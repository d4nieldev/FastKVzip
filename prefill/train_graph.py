"""Train whole-context graph FastKVzip online, one context at a time."""

from __future__ import annotations

import argparse
import copy
import math
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
import wandb
from attention.gate import Weight, load_fastkvzip
from graph import (
    FaissGraphBuilder,
    GraphScorer,
    GraphTrainer,
    PhaseTiming,
    SchedulerSpec,
    TeacherExample,
    build_adamw_optimizers,
    build_scheduler,
    initialize_b_projection,
    load_checkpoint,
    load_gate_checkpoint,
    parse_scheduler_spec,
    resolve_graph_microbatch_size,
    resolve_joint_settings,
    save_checkpoint,
)


TRAIN_KEYS = tuple(
    [("fineweb_10k", index) for index in range(29)]
    + [("fineweb_10k_cat", index) for index in range(5)]
)
VALIDATION_KEYS = tuple(
    [("fineweb_10k", index) for index in range(29, 32)]
    + [("fineweb_10k_cat", 5)]
)


def _auto_or_int(value: str):
    return "auto" if value == "auto" else int(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("graph_checkpoints"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-contexts", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--prefill-chunk", type=int)

    parser.add_argument("--gate-checkpoint")
    parser.add_argument("--gate-dim", type=int)
    parser.add_argument("--gate-sink", type=int)
    parser.add_argument(
        "--freeze-gate", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--b-init", choices=("auto", "zero", "random"))

    parser.add_argument("--graph-dim", type=int)
    parser.add_argument("--gin-depth", type=int)
    parser.add_argument("--num-neighbors", type=int)
    parser.add_argument("--graph-microbatch-size", type=_auto_or_int)
    parser.add_argument("--token-microbatch-size", type=int)
    parser.add_argument(
        "--knn-index", choices=("ivf-flat", "ivf-pq", "ivf_flat", "ivf_pq")
    )
    parser.add_argument("--ivf-nlist", type=int)
    parser.add_argument("--ivf-nprobe", type=int)
    parser.add_argument("--ivfpq-m", type=_auto_or_int)
    parser.add_argument("--ivfpq-bits", type=int)

    parser.add_argument(
        "--training-mode",
        "--mode",
        dest="mode",
        choices=("two_phase", "two-phase", "joint"),
    )
    parser.add_argument("--gate-lr", type=float)
    parser.add_argument("--graph-lr", type=float)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gate-lr-scheduler")
    parser.add_argument("--gate-lr-scheduler-kwargs")
    parser.add_argument("--graph-lr-scheduler")
    parser.add_argument("--graph-lr-scheduler-kwargs")

    parser.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled"), default="online"
    )
    parser.add_argument("--wandb-project", default="whole-context-graph-fastkvzip")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-name")
    return parser


@dataclass(frozen=True)
class TrainingOptions:
    model_id: str
    output_dir: Path
    epochs: int
    max_contexts: int | None
    seed: int
    resume: Path | None
    prefill_chunk: int
    gate_checkpoint: str | None
    gate_dim: int
    gate_sink: int
    gate_dim_explicit: bool
    gate_sink_explicit: bool
    freeze_gate: bool
    b_init: str
    graph_dim: int
    gin_depth: int
    num_neighbors: int
    graph_microbatch_size: str | int
    token_microbatch_size: int
    knn_index: str
    ivf_nlist: int
    ivf_nprobe: int
    ivfpq_m: int
    ivfpq_bits: int
    mode: str
    gate_lr: float
    graph_lr: float
    weight_decay: float
    gate_scheduler: SchedulerSpec | None
    graph_scheduler: SchedulerSpec | None
    wandb_mode: str
    wandb_project: str
    wandb_entity: str | None
    wandb_name: str | None


def _plain_scheduler(spec: SchedulerSpec | None):
    if spec is None:
        return None
    return {"name": spec.name, "kwargs": copy.deepcopy(spec.kwargs)}


def _saved_scheduler(value) -> SchedulerSpec | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != {"name", "kwargs"}:
        raise ValueError("checkpoint has an invalid scheduler specification")
    return parse_scheduler_spec(value["name"], value["kwargs"])


def _checkpoint_gate_metadata(payload) -> dict[str, int]:
    if not isinstance(payload, Mapping):
        return {}
    config = payload.get("config")
    if isinstance(config, Mapping) and "gate_dim" in config:
        return {
            "gate_dim": int(config["gate_dim"]),
            "gate_sink": int(config["gate_sink"]),
        }
    state = payload.get("gate")
    if isinstance(state, Mapping):
        q_norm = next(
            (value for name, value in state.items() if name.endswith("q_norm.weight")),
            None,
        )
        k_base = next(
            (value for name, value in state.items() if name.endswith("k_base")), None
        )
    else:
        modules = payload.get("module")
        first = modules[0] if isinstance(modules, (list, tuple)) and modules else {}
        q_norm, k_base = first.get("q_norm.weight"), first.get("k_base")
    if q_norm is None or k_base is None:
        return {}
    return {"gate_dim": q_norm.numel(), "gate_sink": k_base.shape[-2]}


def _pick(cli_value, saved, key, default, *, normalize=lambda value: value):
    has_saved = isinstance(saved, Mapping) and key in saved
    if cli_value is None:
        return saved[key] if has_saved else default
    value = normalize(cli_value)
    if has_saved and value != saved[key]:
        raise ValueError(f"resume configuration conflicts for {key}")
    return value


def _scheduler_option(args, prefix: str, saved) -> SchedulerSpec | None:
    name = getattr(args, f"{prefix}_lr_scheduler")
    kwargs = getattr(args, f"{prefix}_lr_scheduler_kwargs")
    saved_value = saved.get(f"{prefix}_lr_scheduler") if saved else None
    if name is None:
        if kwargs is not None:
            raise ValueError("scheduler kwargs were supplied without a scheduler")
        return _saved_scheduler(saved_value)
    spec = parse_scheduler_spec(None if name.lower() == "none" else name, kwargs)
    if saved and _plain_scheduler(spec) != saved_value:
        raise ValueError(f"resume configuration conflicts for {prefix}_lr_scheduler")
    return spec


def _load_payload(path: Path | str | None):
    if path is None:
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def resolve_options(args, resume_payload=None, gate_payload=None) -> TrainingOptions:
    """Validate all model-independent settings before loading the LLM."""

    saved = resume_payload.get("config", {}) if resume_payload else {}
    if resume_payload and resume_payload.get("model_id") != args.model:
        raise ValueError("resume checkpoint model identifier conflicts with --model")
    if args.resume is not None and args.gate_checkpoint is not None:
        raise ValueError("--resume and --gate-checkpoint cannot be combined")
    freeze_gate = bool(_pick(args.freeze_gate, saved, "freeze_gate", False))
    if freeze_gate and args.gate_checkpoint is None and args.resume is None:
        raise ValueError("--freeze-gate requires --gate-checkpoint or --resume")
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    if args.max_contexts is not None and args.max_contexts < 1:
        raise ValueError("max-contexts must be positive")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError("weight decay must be finite and non-negative")

    gate_metadata = _checkpoint_gate_metadata(gate_payload)
    if (
        args.gate_dim is not None
        and "gate_dim" in gate_metadata
        and args.gate_dim != gate_metadata["gate_dim"]
    ):
        raise ValueError("--gate-dim conflicts with the gate checkpoint")
    if (
        args.gate_sink is not None
        and "gate_sink" in gate_metadata
        and args.gate_sink != gate_metadata["gate_sink"]
    ):
        raise ValueError("--gate-sink conflicts with the gate checkpoint")
    gate_default = gate_metadata.get("gate_dim", 16)
    sink_default = gate_metadata.get("gate_sink", 16)
    gate_dim = int(_pick(args.gate_dim, saved, "gate_dim", gate_default))
    gate_sink = int(_pick(args.gate_sink, saved, "gate_sink", sink_default))
    graph_dim = int(_pick(args.graph_dim, saved, "graph_dim", 32))
    gin_depth = int(_pick(args.gin_depth, saved, "gin_depth", 1))
    num_neighbors = int(_pick(args.num_neighbors, saved, "num_neighbors", 16))
    token_microbatch_size = int(
        _pick(args.token_microbatch_size, saved, "token_microbatch_size", 1000)
    )
    ivf_nlist = int(_pick(args.ivf_nlist, saved, "ivf_nlist", 256))
    ivf_nprobe = int(_pick(args.ivf_nprobe, saved, "ivf_nprobe", 16))
    ivfpq_bits = int(_pick(args.ivfpq_bits, saved, "ivfpq_bits", 8))
    for name, value in (
        ("gate dim", gate_dim),
        ("gate sink", gate_sink),
        ("graph dim", graph_dim),
        ("GIN depth", gin_depth),
        ("neighbors", num_neighbors),
        ("token microbatch size", token_microbatch_size),
        ("IVF nlist", ivf_nlist),
        ("IVF nprobe", ivf_nprobe),
        ("IVF-PQ bits", ivfpq_bits),
    ):
        if value < 1:
            raise ValueError(f"{name} must be positive")

    knn_index = _pick(
        args.knn_index,
        saved,
        "knn_index",
        "ivf_flat",
        normalize=lambda value: value.replace("-", "_"),
    )
    graph_microbatch_cli = args.graph_microbatch_size
    if saved and graph_microbatch_cli == "auto":
        graph_microbatch_cli = None
    graph_microbatch_size = _pick(
        graph_microbatch_cli, saved, "graph_microbatch_size", "auto"
    )
    pq_cli = args.ivfpq_m
    pq_default = FaissGraphBuilder.auto_pq_m(graph_dim)
    ivfpq_m = int(
        _pick(
            pq_cli,
            saved,
            "ivfpq_m",
            pq_default,
            normalize=lambda value: pq_default if value == "auto" else int(value),
        )
    )
    if ivfpq_m < 1:
        raise ValueError("ivfpq-m must be positive")
    if graph_dim % ivfpq_m:
        raise ValueError("ivfpq-m must divide graph-dim")

    mode = _pick(
        args.mode,
        saved,
        "training_mode",
        "two_phase",
        normalize=lambda value: value.replace("-", "_"),
    )
    if mode not in {"two_phase", "joint"}:
        raise ValueError("training mode must be two_phase or joint")
    gate_scheduler = _scheduler_option(args, "gate", saved)
    graph_scheduler = _scheduler_option(args, "graph", saved)
    gate_lr = args.gate_lr
    graph_lr = args.graph_lr
    if saved:
        gate_lr = _pick(gate_lr, saved, "gate_lr", None)
        graph_lr = _pick(graph_lr, saved, "graph_lr", None)
    if mode == "joint":
        both_scheduler_flags = (
            args.gate_lr_scheduler is not None
            and args.graph_lr_scheduler is not None
        )
        if (
            not freeze_gate
            and both_scheduler_flags
            and gate_scheduler != graph_scheduler
        ):
            raise ValueError("joint schedulers must be equal")
        gate_lr, graph_lr, gate_scheduler, graph_scheduler = resolve_joint_settings(
            gate_lr,
            graph_lr,
            gate_scheduler,
            graph_scheduler,
            gate_frozen=freeze_gate,
        )
    else:
        gate_lr = 1e-4 if gate_lr is None else gate_lr
        graph_lr = 1e-3 if graph_lr is None else graph_lr
        for value in (gate_lr, graph_lr):
            if (
                not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError("learning rates must be finite and positive")

    b_init_cli = args.b_init
    if saved and b_init_cli == "auto":
        b_init_cli = None
    b_init = _pick(b_init_cli, saved, "b_init", "auto")
    prefill_chunk = int(
        _pick(
            args.prefill_chunk,
            {"prefill_chunk": resume_payload["prefill_chunk"]}
            if resume_payload
            else {},
            "prefill_chunk",
            16000,
        )
    )
    if prefill_chunk < 1:
        raise ValueError("prefill chunk must be positive")

    return TrainingOptions(
        model_id=args.model,
        output_dir=args.output_dir,
        epochs=args.epochs,
        max_contexts=args.max_contexts,
        seed=args.seed,
        resume=args.resume,
        prefill_chunk=prefill_chunk,
        gate_checkpoint=args.gate_checkpoint,
        gate_dim=gate_dim,
        gate_sink=gate_sink,
        gate_dim_explicit=args.gate_dim is not None,
        gate_sink_explicit=args.gate_sink is not None,
        freeze_gate=freeze_gate,
        b_init=b_init,
        graph_dim=graph_dim,
        gin_depth=gin_depth,
        num_neighbors=num_neighbors,
        graph_microbatch_size=graph_microbatch_size,
        token_microbatch_size=token_microbatch_size,
        knn_index=knn_index,
        ivf_nlist=ivf_nlist,
        ivf_nprobe=ivf_nprobe,
        ivfpq_m=ivfpq_m,
        ivfpq_bits=ivfpq_bits,
        mode=mode,
        gate_lr=float(gate_lr),
        graph_lr=float(graph_lr),
        weight_decay=args.weight_decay,
        gate_scheduler=gate_scheduler,
        graph_scheduler=graph_scheduler,
        wandb_mode=args.wandb_mode,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_name=args.wandb_name,
    )


def build_teacher(model_id: str, *, model_factory=None):
    if model_factory is None:
        from model import ModelKVzip

        model_factory = ModelKVzip
    return model_factory(model_id, kv_type="retain", gate_path_or_name="")


def teacher_example_from_kv(kv, dataset_name: str, dataset_index: int) -> TeacherExample:
    start_idx, end_idx = int(kv.start_idx), int(kv.end_idx)
    sequence_length = end_idx - start_idx
    hidden = []
    for layer_hidden in kv.hidden_cache:
        if layer_hidden.ndim != 3 or layer_hidden.size(0) != 1:
            raise ValueError("cached hidden states must have shape [1,prefix+tokens,dim]")
        hidden.append(layer_hidden[:, start_idx:end_idx, :])
    scores = (
        torch.stack(kv.score, dim=0)
        if isinstance(kv.score, (list, tuple))
        else kv.score
    )
    if scores.ndim != 4:
        raise ValueError("teacher scores must have shape [layers,1,heads,tokens]")
    if scores.size(-1) == end_idx:
        scores = scores[..., start_idx:end_idx]
    if scores.size(-1) != sequence_length:
        raise ValueError("teacher score length does not match context")
    return TeacherExample(
        dataset_name=dataset_name,
        dataset_index=dataset_index,
        token_ids=kv.prefill_ids[:, start_idx:end_idx],
        hidden_by_layer=hidden,
        teacher_scores=scores,
        prefix_ids=kv.prefill_ids[:, :start_idx],
        sequence_length=sequence_length,
    )


def initial_cursor() -> dict[str, object]:
    return {
        "epoch": 0,
        "phase": "train",
        "offset": 0,
        "validation_sum": 0.0,
        "validation_count": 0,
        "best_validation_bce": float("inf"),
        "wandb_step": 0,
    }


def next_context_key(cursor) -> tuple[str, int]:
    keys = TRAIN_KEYS if cursor["phase"] == "train" else VALIDATION_KEYS
    return keys[cursor["offset"]]


def advance_cursor(cursor, *, validation_loss: float | None = None):
    """Advance a next-item cursor and return a completed validation mean, if any."""

    cursor = copy.deepcopy(cursor)
    phase = cursor["phase"]
    if phase not in {"train", "validation"}:
        raise ValueError("cursor phase must be train or validation")
    if phase == "validation":
        if validation_loss is None or not math.isfinite(validation_loss):
            raise ValueError("validation cursor advancement requires a finite loss")
        cursor["validation_sum"] += validation_loss
        cursor["validation_count"] += 1
    elif validation_loss is not None:
        raise ValueError("training cursor does not accept validation loss")
    cursor["offset"] += 1
    cursor["wandb_step"] += 1

    completed_mean = None
    keys = TRAIN_KEYS if phase == "train" else VALIDATION_KEYS
    if cursor["offset"] == len(keys):
        cursor["offset"] = 0
        if phase == "train":
            cursor["phase"] = "validation"
        else:
            completed_mean = cursor["validation_sum"] / cursor["validation_count"]
            cursor["best_validation_bce"] = min(
                cursor["best_validation_bce"], completed_mean
            )
            cursor["epoch"] += 1
            cursor["phase"] = "train"
            cursor["validation_sum"] = 0.0
            cursor["validation_count"] = 0
    return cursor, completed_mean


def _model_dimensions(teacher):
    config = getattr(teacher.config, "text_config", teacher.config)
    layers = int(config.num_hidden_layers)
    kv_heads = int(config.num_key_value_heads)
    query_heads = int(config.num_attention_heads)
    if query_heads % kv_heads:
        raise ValueError("attention heads must be divisible by KV heads")
    return config, layers, kv_heads, query_heads // kv_heads


def _random_gates(teacher, config, options: TrainingOptions):
    return [
        Weight(
            index=layer,
            input_dim=config.hidden_size,
            output_dim=options.gate_dim,
            nhead=config.num_key_value_heads,
            ngroup=config.num_attention_heads // config.num_key_value_heads,
            dtype=teacher.dtype,
            sink=options.gate_sink,
        ).to(teacher.device)
        for layer in range(config.num_hidden_layers)
    ]


def _student_gates(teacher, config, options: TrainingOptions):
    if options.gate_checkpoint == "fastkvzip":
        model_name = getattr(teacher.model, "name_or_path", options.model_id)
        gates = load_fastkvzip(model_name, "fastkvzip", device=teacher.device)
        actual_dim, actual_sink = gates[0].output_dim, gates[0].sink
        if options.gate_dim_explicit and options.gate_dim != actual_dim:
            raise ValueError("--gate-dim conflicts with the FastKVzip checkpoint")
        if options.gate_sink_explicit and options.gate_sink != actual_sink:
            raise ValueError("--gate-sink conflicts with the FastKVzip checkpoint")
        return gates, replace(options, gate_dim=actual_dim, gate_sink=actual_sink)
    return _random_gates(teacher, config, options), options


def normalized_checkpoint_config(
    *, model_id: str, scorer: GraphScorer, options, query_groups: int
) -> dict[str, object]:
    return {
        "format_version": 1,
        "model_id": model_id,
        "gate_dim": scorer.gate_dim,
        "gate_sink": scorer.gates[0].sink,
        "hidden_dim": scorer.hidden_dim,
        "num_layers": scorer.num_layers,
        "num_kv_heads": scorer.num_heads,
        "query_groups": query_groups,
        "graph_dim": options.graph_dim,
        "gin_depth": options.gin_depth,
        "graph_microbatch_size": options.graph_microbatch_size,
        "num_neighbors": options.num_neighbors,
        "knn_index": options.knn_index,
        "ivf_nlist": options.ivf_nlist,
        "ivf_nprobe": options.ivf_nprobe,
        "ivfpq_m": options.ivfpq_m,
        "ivfpq_bits": options.ivfpq_bits,
        "training_mode": options.mode,
        "token_microbatch_size": options.token_microbatch_size,
        "gate_lr": options.gate_lr,
        "graph_lr": options.graph_lr,
        "gate_lr_scheduler": _plain_scheduler(options.gate_scheduler),
        "graph_lr_scheduler": _plain_scheduler(options.graph_scheduler),
        "b_init": options.b_init,
        "freeze_gate": options.freeze_gate,
    }


def _validate_resume_config(saved, current) -> None:
    if saved != current:
        differing = sorted(
            key for key in set(saved) | set(current) if saved.get(key) != current.get(key)
        )
        raise ValueError(f"resume configuration conflicts for: {', '.join(differing)}")


def _initialize_wandb(options: TrainingOptions, module, *, run_id=None):
    if options.wandb_mode == "online":
        logged_in = module.login()
        if logged_in is False:
            raise RuntimeError("Weights & Biases login failed")
    kwargs = {
        "project": options.wandb_project,
        "entity": options.wandb_entity,
        "name": options.wandb_name,
        "mode": options.wandb_mode,
    }
    if run_id is not None:
        kwargs.update(id=run_id, resume="allow")
    return module.init(**kwargs)


def _optimizer_lr(optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def _reset_peak_memory_stats(device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def _materialize_context_result(result) -> dict[str, object]:
    materialized = dict(result)
    for key in ("gate_loss", "graph_loss", "delta_energy_share"):
        value = materialized[key]
        if isinstance(value, torch.Tensor):
            materialized[key] = float(value.detach().item())
    return materialized


def run_and_log_context(
    trainer: GraphTrainer,
    example: TeacherExample,
    *,
    mode: str,
    validation: bool,
    run,
    step: int,
    reset_memory_stats: bool = True,
):
    """Run and log one context; this is the sole W&B metric emission point."""

    device = trainer.scorer.a_proj.weight.device
    use_cuda = device.type == "cuda" and torch.cuda.is_available()
    if reset_memory_stats:
        _reset_peak_memory_stats(device)
    timing = PhaseTiming(device)
    previous_timing = trainer.timing
    trainer.timing = timing
    try:
        if validation:
            validation_result = trainer.evaluate_context(example)
            result = {
                "gate_loss": None,
                "graph_loss": validation_result.loss,
                "delta_energy_share": validation_result.delta_energy_share,
                "gate_steps": 0,
                "graph_steps": 0,
            }
        else:
            result = trainer.train_context(example, mode=mode)
    finally:
        trainer.timing = previous_timing
    elapsed = timing.resolve()
    result = _materialize_context_result(result)
    metrics = {}
    if result["gate_loss"] is not None:
        metrics["gate/bce"] = result["gate_loss"]
    if result["graph_loss"] is not None:
        metrics["graph/bce"] = result["graph_loss"]
    metrics["delta_energy_share"] = result["delta_energy_share"]
    metrics.update({key.replace("_", "/", 1): value for key, value in elapsed.items()})
    metrics["gpu/peak_allocated_bytes"] = (
        torch.cuda.max_memory_allocated(device) if use_cuda else 0
    )
    metrics["gpu/peak_reserved_bytes"] = (
        torch.cuda.max_memory_reserved(device) if use_cuda else 0
    )
    if trainer.gate_optimizer is not None:
        metrics["gate/learning_rate"] = _optimizer_lr(trainer.gate_optimizer)
    if trainer.graph_optimizer is not None:
        metrics["graph/learning_rate"] = _optimizer_lr(trainer.graph_optimizer)
    for optimizer in (trainer.gate_optimizer, trainer.graph_optimizer):
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
    run.log(metrics, step=step)
    return result, metrics


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_components(teacher, options, resume_payload):
    config, layers, heads, query_groups = _model_dimensions(teacher)
    microbatch = resolve_graph_microbatch_size(
        options.graph_microbatch_size, layers, heads
    )
    options = replace(options, graph_microbatch_size=microbatch)
    gates, options = _student_gates(teacher, config, options)
    builder = FaissGraphBuilder(
        k=options.num_neighbors,
        index_mode=options.knn_index,
        nlist=options.ivf_nlist,
        nprobe=options.ivf_nprobe,
        pq_m=options.ivfpq_m,
        pq_bits=options.ivfpq_bits,
    )
    scorer = GraphScorer(
        gates,
        teacher.config,
        graph_dim=options.graph_dim,
        gin_depth=options.gin_depth,
        graph_builder=builder,
        graph_microbatch_size=microbatch,
    )
    has_gate_checkpoint = (
        options.gate_checkpoint is not None or resume_payload is not None
    )
    if resume_payload is None:
        if options.gate_checkpoint not in {None, "fastkvzip"}:
            load_gate_checkpoint(scorer, options.gate_checkpoint)
        resolved_b_init = initialize_b_projection(
            scorer, options.b_init, has_gate_checkpoint=has_gate_checkpoint
        )
        options = replace(options, b_init=resolved_b_init)

    gate_frozen = options.freeze_gate
    gate_optimizer, graph_optimizer = build_adamw_optimizers(
        scorer,
        gate_lr=options.gate_lr,
        graph_lr=options.graph_lr,
        weight_decay=options.weight_decay,
        gate_frozen=gate_frozen,
        graph_frozen=False,
    )
    gate_scheduler = (
        build_scheduler(gate_optimizer, options.gate_scheduler)
        if gate_optimizer is not None
        else None
    )
    graph_scheduler = (
        build_scheduler(graph_optimizer, options.graph_scheduler)
        if graph_optimizer is not None
        else None
    )
    trainer = GraphTrainer(
        scorer,
        gate_optimizer=gate_optimizer,
        graph_optimizer=graph_optimizer,
        gate_scheduler=gate_scheduler,
        graph_scheduler=graph_scheduler,
        token_microbatch_size=options.token_microbatch_size,
        graph_microbatch_size=microbatch,
    )
    checkpoint_config = normalized_checkpoint_config(
        model_id=options.model_id,
        scorer=scorer,
        options=options,
        query_groups=query_groups,
    )
    return options, scorer, trainer, checkpoint_config


def run_training(
    args,
    *,
    model_factory=None,
    dataset_loader=None,
    wrapper_factory=None,
    wandb_module=wandb,
):
    resume_payload = _load_payload(args.resume)
    gate_payload = None
    if args.gate_checkpoint not in {None, "fastkvzip"}:
        gate_payload = _load_payload(args.gate_checkpoint)
    options = resolve_options(args, resume_payload, gate_payload)
    del gate_payload
    resume_run_id = (
        resume_payload.get("wandb_run_id") if resume_payload is not None else None
    )
    run = _initialize_wandb(
        options,
        wandb_module,
        run_id=resume_run_id,
    )
    try:
        _set_seed(options.seed)
        teacher = build_teacher(options.model_id, model_factory=model_factory)
        # This model-dependent check intentionally precedes gates, datasets, and prefill.
        _, layers, heads, _ = _model_dimensions(teacher)
        resolve_graph_microbatch_size(options.graph_microbatch_size, layers, heads)
        if dataset_loader is None or wrapper_factory is None:
            from data import DataWrapper, load_dataset_all

            dataset_loader = dataset_loader or load_dataset_all
            wrapper_factory = wrapper_factory or DataWrapper
        options, scorer, trainer, checkpoint_config = _make_components(
            teacher, options, resume_payload
        )
        if resume_payload is not None:
            _validate_resume_config(resume_payload["config"], checkpoint_config)
            load_checkpoint(
                resume_payload,
                scorer=scorer,
                gate_optimizer=trainer.gate_optimizer,
                graph_optimizer=trainer.graph_optimizer,
                gate_scheduler=trainer.gate_scheduler,
                graph_scheduler=trainer.graph_scheduler,
            )
            cursor = copy.deepcopy(resume_payload["data_cursor"])
            training_prefix = resume_payload["prefix_ids"].detach().to("cpu").clone()
            del resume_payload
        else:
            cursor = initial_cursor()
            training_prefix = None
        if hasattr(run, "config"):
            run.config.update(checkpoint_config, allow_val_change=True)

        wrappers = {}

        def make_example(key):
            nonlocal training_prefix
            dataset_name, dataset_index = key
            if dataset_name not in wrappers:
                dataset = dataset_loader(dataset_name, teacher.tokenizer)
                wrappers[dataset_name] = wrapper_factory(dataset_name, dataset, teacher)
            if training_prefix is not None:
                teacher.sys_prompt_ids = training_prefix.to(teacher.device)
            kv = wrappers[dataset_name].prefill_context(
                dataset_index,
                prefill_chunk=options.prefill_chunk,
                save_hidden=True,
                do_score=True,
            )
            example = teacher_example_from_kv(kv, dataset_name, dataset_index)
            del kv
            if training_prefix is None:
                training_prefix = example.prefix_ids
            elif not torch.equal(example.prefix_ids, training_prefix):
                raise ValueError("context prefix differs from the checkpointed prefix")
            return example

        def save(kind):
            save_checkpoint(
                options.output_dir,
                kind,
                scorer=scorer,
                config=checkpoint_config,
                model_id=options.model_id,
                prefix_ids=training_prefix,
                prefill_chunk=options.prefill_chunk,
                data_cursor=cursor,
                wandb_run_id=getattr(run, "id", None),
                gate_optimizer=trainer.gate_optimizer,
                graph_optimizer=trainer.graph_optimizer,
                gate_scheduler=trainer.gate_scheduler,
                graph_scheduler=trainer.graph_scheduler,
            )

        processed_contexts = 0
        while cursor["epoch"] < options.epochs and (
            options.max_contexts is None
            or processed_contexts < options.max_contexts
        ):
            validation = cursor["phase"] == "validation"
            scorer_device = scorer.a_proj.weight.device
            # Include online teacher generation and student training in one context peak.
            _reset_peak_memory_stats(scorer_device)
            example = make_example(next_context_key(cursor))
            result, _ = run_and_log_context(
                trainer,
                example,
                mode=options.mode,
                validation=validation,
                run=run,
                step=cursor["wandb_step"],
                reset_memory_stats=False,
            )
            del example
            previous_best = cursor["best_validation_bce"]
            cursor, validation_mean = advance_cursor(
                cursor,
                validation_loss=result["graph_loss"] if validation else None,
            )
            if validation_mean is not None:
                trainer.step_validation(validation_mean)
                if validation_mean < previous_best:
                    save("best")
            save("last")
            processed_contexts += 1
        return options.output_dir / "last.pt"
    finally:
        run.finish()


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    run_training(args)


if __name__ == "__main__":
    main()
