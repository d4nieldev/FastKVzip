"""Train the streamed implicit whole-context FastKVzip mixer."""

from __future__ import annotations

import argparse
import copy
import math
import os
import random
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
import wandb
from attention.gate import Weight, load_fastkvzip
from graph import (
    GraphTrainer,
    ImplicitGraphScorer,
    PhaseTiming,
    SchedulerSpec,
    TeacherExample,
    build_adamw_optimizers,
    build_scheduler,
    compute_dtype_name,
    load_checkpoint,
    load_gate_checkpoint,
    parse_compute_dtype,
    parse_scheduler_spec,
    resolve_graph_microbatch_size,
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
    parser.add_argument("--save-strategy", choices=("epochs", "steps"), default="epochs")
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--eval-strategy", choices=("epochs", "steps"), default="epochs")
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--prefill-chunk", type=int)
    parser.add_argument("--teacher-cache-dir", type=Path)

    parser.add_argument("--gate-checkpoint")
    parser.add_argument("--gate-dim", type=int)
    parser.add_argument("--gate-sink", type=int)
    parser.add_argument(
        "--freeze-gate", action=argparse.BooleanOptionalAction, default=None
    )

    parser.add_argument("--graph-dim", type=int)
    parser.add_argument("--gram-normalization", choices=("token-count", "none"))
    parser.add_argument("--leaky-relu-slope", type=float)
    parser.add_argument("--alpha-init", type=float)
    parser.add_argument("--graph-microbatch-size", type=_auto_or_int)
    parser.add_argument("--token-microbatch-size", type=int)

    parser.add_argument(
        "--training-mode",
        "--mode",
        dest="mode",
        choices=("two_phase", "two-phase", "joint"),
    )
    parser.add_argument("--gate-lr", type=float)
    parser.add_argument("--mixer-lr", "--graph-lr", dest="mixer_lr", type=float)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gate-lr-scheduler")
    parser.add_argument("--gate-lr-scheduler-kwargs")
    parser.add_argument(
        "--mixer-lr-scheduler", "--graph-lr-scheduler", dest="mixer_lr_scheduler"
    )
    parser.add_argument(
        "--mixer-lr-scheduler-kwargs",
        "--graph-lr-scheduler-kwargs",
        dest="mixer_lr_scheduler_kwargs",
    )

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
    save_strategy: str
    save_every: int
    eval_strategy: str
    eval_every: int
    seed: int
    resume: Path | None
    prefill_chunk: int
    teacher_cache_dir: Path | None
    gate_checkpoint: str | None
    gate_dim: int
    gate_sink: int
    gate_dim_explicit: bool
    gate_sink_explicit: bool
    freeze_gate: bool
    compute_dtype: str | None
    graph_dim: int
    gram_normalization: str
    leaky_relu_slope: float
    alpha_init: float
    graph_microbatch_size: str | int
    token_microbatch_size: int
    mode: str
    gate_lr: float
    mixer_lr: float
    weight_decay: float
    gate_scheduler: SchedulerSpec | None
    mixer_scheduler: SchedulerSpec | None
    wandb_mode: str
    wandb_project: str
    wandb_entity: str | None
    wandb_name: str | None


def _plain_scheduler(spec: SchedulerSpec | None):
    return None if spec is None else {"name": spec.name, "kwargs": copy.deepcopy(spec.kwargs)}


def _saved_scheduler(value) -> SchedulerSpec | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != {"name", "kwargs"}:
        raise ValueError("checkpoint has an invalid scheduler specification")
    return parse_scheduler_spec(value["name"], value["kwargs"])


def _checkpoint_gate_metadata(payload) -> dict[str, object]:
    if not isinstance(payload, Mapping):
        return {}
    metadata = {}
    config = payload.get("config")
    if isinstance(config, Mapping) and "gate_dim" in config:
        metadata.update(gate_dim=int(config["gate_dim"]), gate_sink=int(config["gate_sink"]))
    if isinstance(config, Mapping) and "compute_dtype" in config:
        parse_compute_dtype(config["compute_dtype"])
        metadata["compute_dtype"] = config["compute_dtype"]
    state = payload.get("gate")
    if isinstance(state, Mapping):
        q_norm = next((value for name, value in state.items() if name.endswith("q_norm.weight")), None)
        k_base = next((value for name, value in state.items() if name.endswith("k_base")), None)
        q_proj = next((value for name, value in state.items() if name.endswith("q_proj.weight")), None)
    else:
        modules = payload.get("module")
        first = modules[0] if isinstance(modules, (list, tuple)) and modules else {}
        q_norm, k_base = first.get("q_norm.weight"), first.get("k_base")
        q_proj = first.get("q_proj.weight")
    if q_norm is not None and k_base is not None:
        metadata.setdefault("gate_dim", q_norm.numel())
        metadata.setdefault("gate_sink", k_base.shape[-2])
    if "compute_dtype" not in metadata and isinstance(q_proj, torch.Tensor):
        metadata["compute_dtype"] = compute_dtype_name(q_proj.dtype)
    return metadata


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
    return None if path is None else torch.load(path, map_location="cpu", weights_only=False)


def _positive_finite(name: str, value: float, *, allow_zero: bool = False) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        or (not allow_zero and value == 0)
    ):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return float(value)


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def resolve_options(args, resume_payload=None, gate_payload=None) -> TrainingOptions:
    """Validate model-independent settings before loading the LLM."""

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
    save_every = _positive_int("save-every", args.save_every)
    eval_every = _positive_int("eval-every", args.eval_every)
    weight_decay = _positive_finite("weight decay", args.weight_decay, allow_zero=True)

    gate_metadata = _checkpoint_gate_metadata(gate_payload)
    compute_dtype = saved.get("compute_dtype", gate_metadata.get("compute_dtype"))
    if compute_dtype is not None:
        parse_compute_dtype(compute_dtype)
    if args.gate_dim is not None and "gate_dim" in gate_metadata and args.gate_dim != gate_metadata["gate_dim"]:
        raise ValueError("--gate-dim conflicts with the gate checkpoint")
    if args.gate_sink is not None and "gate_sink" in gate_metadata and args.gate_sink != gate_metadata["gate_sink"]:
        raise ValueError("--gate-sink conflicts with the gate checkpoint")
    gate_dim = int(_pick(args.gate_dim, saved, "gate_dim", gate_metadata.get("gate_dim", 16)))
    gate_sink = int(_pick(args.gate_sink, saved, "gate_sink", gate_metadata.get("gate_sink", 16)))
    graph_dim = int(_pick(args.graph_dim, saved, "graph_dim", 32))
    token_microbatch_size = int(
        _pick(args.token_microbatch_size, saved, "token_microbatch_size", 1000)
    )
    for name, value in (
        ("gate dim", gate_dim),
        ("gate sink", gate_sink),
        ("graph dim", graph_dim),
        ("token microbatch size", token_microbatch_size),
    ):
        if value < 1:
            raise ValueError(f"{name} must be positive")
    gram_normalization = _pick(
        args.gram_normalization, saved, "gram_normalization", "token-count"
    )
    if gram_normalization not in {"token-count", "none"}:
        raise ValueError("gram normalization must be token-count or none")
    leaky_relu_slope = _positive_finite(
        "leaky ReLU slope",
        _pick(args.leaky_relu_slope, saved, "leaky_relu_slope", 0.01),
        allow_zero=True,
    )
    alpha_init = _pick(args.alpha_init, saved, "alpha_init", 0.1)
    if isinstance(alpha_init, bool) or not isinstance(alpha_init, (int, float)) or not math.isfinite(alpha_init):
        raise ValueError("alpha init must be finite")

    graph_microbatch_cli = args.graph_microbatch_size
    if saved and graph_microbatch_cli == "auto":
        graph_microbatch_cli = None
    graph_microbatch_size = _pick(
        graph_microbatch_cli, saved, "graph_microbatch_size", "auto"
    )
    mode = _pick(
        args.mode, saved, "training_mode", "joint", normalize=lambda value: value.replace("-", "_")
    )
    if mode not in {"two_phase", "joint"}:
        raise ValueError("training mode must be two_phase or joint")
    gate_scheduler = _scheduler_option(args, "gate", saved)
    mixer_scheduler = _scheduler_option(args, "mixer", saved)
    gate_lr = _positive_finite(
        "gate learning rate", _pick(args.gate_lr, saved, "gate_lr", 1e-4)
    )
    mixer_lr = _positive_finite(
        "mixer learning rate", _pick(args.mixer_lr, saved, "mixer_lr", 1e-3)
    )
    prefill_chunk = int(
        _pick(
            args.prefill_chunk,
            {"prefill_chunk": resume_payload["prefill_chunk"]} if resume_payload else {},
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
        save_strategy=args.save_strategy,
        save_every=save_every,
        eval_strategy=args.eval_strategy,
        eval_every=eval_every,
        seed=args.seed,
        resume=args.resume,
        prefill_chunk=prefill_chunk,
        teacher_cache_dir=args.teacher_cache_dir,
        gate_checkpoint=args.gate_checkpoint,
        gate_dim=gate_dim,
        gate_sink=gate_sink,
        gate_dim_explicit=args.gate_dim is not None,
        gate_sink_explicit=args.gate_sink is not None,
        freeze_gate=freeze_gate,
        compute_dtype=compute_dtype,
        graph_dim=graph_dim,
        gram_normalization=gram_normalization,
        leaky_relu_slope=leaky_relu_slope,
        alpha_init=float(alpha_init),
        graph_microbatch_size=graph_microbatch_size,
        token_microbatch_size=token_microbatch_size,
        mode=mode,
        gate_lr=gate_lr,
        mixer_lr=mixer_lr,
        weight_decay=weight_decay,
        gate_scheduler=gate_scheduler,
        mixer_scheduler=mixer_scheduler,
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


def _normal_capture(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.device.type == "cpu" and not tensor.requires_grad and not tensor.is_inference():
        return tensor
    with torch.inference_mode(False):
        return tensor.detach().to("cpu", copy=True)


def teacher_example_from_kv(kv, dataset_name: str, dataset_index: int) -> TeacherExample:
    """Transfer retained normal CPU hidden tensors to a teacher example without cloning."""

    start_idx, end_idx = int(kv.start_idx), int(kv.end_idx)
    sequence_length = end_idx - start_idx
    hidden = []
    for layer_hidden in kv.hidden_cache:
        if layer_hidden.ndim != 3 or layer_hidden.size(0) != 1:
            raise ValueError("cached hidden states must have shape [1,prefix+tokens,dim]")
        normal = _normal_capture(layer_hidden)
        hidden.append(normal[:, start_idx:end_idx, :])
    scores = torch.stack(kv.score, dim=0) if isinstance(kv.score, (list, tuple)) else kv.score
    if scores.ndim != 4:
        raise ValueError("teacher scores must have shape [layers,1,heads,tokens]")
    if scores.size(-1) == end_idx:
        scores = scores[..., start_idx:end_idx]
    if scores.size(-1) != sequence_length:
        raise ValueError("teacher score length does not match context")
    return TeacherExample.from_owned_cpu(
        dataset_name=dataset_name,
        dataset_index=dataset_index,
        token_ids=_normal_capture(kv.prefill_ids[:, start_idx:end_idx]),
        hidden_by_layer=hidden,
        teacher_scores=_normal_capture(scores),
        prefix_ids=_normal_capture(kv.prefill_ids[:, :start_idx]),
        sequence_length=sequence_length,
    )


def _teacher_cache_path(cache_dir: Path, key: tuple[str, int]) -> Path:
    dataset_name, dataset_index = key
    if Path(dataset_name).name != dataset_name:
        raise ValueError("teacher cache dataset name is not a safe path component")
    return cache_dir / dataset_name / f"{dataset_index}.pt"


def _teacher_cache_payload(
    example: TeacherExample, *, model_id: str, prefill_chunk: int
) -> dict[str, object]:
    return {
        "dataset_name": example.dataset_name,
        "dataset_index": example.dataset_index,
        "token_ids": example.token_ids,
        "hidden_by_layer": list(example.hidden_by_layer),
        "teacher_scores": example.teacher_scores,
        "prefix_ids": example.prefix_ids,
        "sequence_length": example.sequence_length,
        "model_id": model_id,
        "prefill_chunk": prefill_chunk,
    }


def _load_teacher_cache(
    path: Path,
    *,
    key: tuple[str, int],
    model_id: str,
    prefill_chunk: int,
) -> TeacherExample:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        required = {
            "dataset_name",
            "dataset_index",
            "token_ids",
            "hidden_by_layer",
            "teacher_scores",
            "prefix_ids",
            "sequence_length",
            "model_id",
            "prefill_chunk",
        }
        if not isinstance(payload, Mapping) or set(payload) != required:
            raise ValueError("cache payload fields do not match")
        if (payload["dataset_name"], payload["dataset_index"]) != key:
            raise ValueError("cache context identity does not match")
        if payload["model_id"] != model_id or payload["prefill_chunk"] != prefill_chunk:
            raise ValueError("cache model or prefill settings do not match")
        return TeacherExample.from_owned_cpu(
            dataset_name=payload["dataset_name"],
            dataset_index=payload["dataset_index"],
            token_ids=payload["token_ids"],
            hidden_by_layer=payload["hidden_by_layer"],
            teacher_scores=payload["teacher_scores"],
            prefix_ids=payload["prefix_ids"],
            sequence_length=payload["sequence_length"],
        )
    except Exception as error:
        raise ValueError(f"invalid or incompatible teacher cache {path}: {error}") from error


def _save_teacher_cache_if_missing(
    path: Path,
    example: TeacherExample,
    *,
    model_id: str,
    prefill_chunk: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"teacher cache already exists: {path}")
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        torch.save(
            _teacher_cache_payload(example, model_id=model_id, prefill_chunk=prefill_chunk),
            temporary_path,
        )
        os.link(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def initial_cursor() -> dict[str, object]:
    return {
        "epoch": 0,
        "phase": "train",
        "offset": 0,
        "best_validation_bce": float("inf"),
        "wandb_step": 0,
    }


def advance_train_cursor(cursor):
    """Advance one whole training context and report epoch completion."""

    cursor = copy.deepcopy(cursor)
    if cursor.get("phase") != "train":
        raise ValueError("checkpoint cursor must be at a training context")
    cursor["offset"] += 1
    cursor["wandb_step"] += 1
    completed_epoch = cursor["offset"] == len(TRAIN_KEYS)
    if completed_epoch:
        cursor["offset"] = 0
        cursor["epoch"] += 1
    return cursor, completed_epoch


def training_context_steps(cursor) -> int:
    return int(cursor["epoch"]) * len(TRAIN_KEYS) + int(cursor["offset"])


def cadence_due(strategy: str, every: int, *, train_steps: int, completed_epoch: bool) -> bool:
    if strategy == "steps":
        return train_steps % every == 0
    return completed_epoch and (train_steps // len(TRAIN_KEYS)) % every == 0


def _model_dimensions(teacher):
    config = getattr(teacher.config, "text_config", teacher.config)
    layers = int(config.num_hidden_layers)
    kv_heads = int(config.num_key_value_heads)
    query_heads = int(config.num_attention_heads)
    if query_heads % kv_heads:
        raise ValueError("attention heads must be divisible by KV heads")
    return config, layers, kv_heads, query_heads // kv_heads


def _random_gates(teacher, config, options: TrainingOptions):
    dtype = teacher.dtype if options.compute_dtype is None else parse_compute_dtype(options.compute_dtype)
    return [
        Weight(
            index=layer,
            input_dim=config.hidden_size,
            output_dim=options.gate_dim,
            nhead=config.num_key_value_heads,
            ngroup=config.num_attention_heads // config.num_key_value_heads,
            dtype=dtype,
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
    *, model_id: str, scorer: ImplicitGraphScorer, options, query_groups: int
) -> dict[str, object]:
    return {
        "model_id": model_id,
        "compute_dtype": compute_dtype_name(scorer.compute_dtype),
        "gate_dim": scorer.gate_dim,
        "gate_sink": scorer.gates[0].sink,
        "hidden_dim": scorer.hidden_dim,
        "num_layers": scorer.num_layers,
        "num_kv_heads": scorer.num_heads,
        "query_groups": query_groups,
        "graph_dim": options.graph_dim,
        "gram_normalization": options.gram_normalization,
        "leaky_relu_slope": options.leaky_relu_slope,
        "alpha_init": options.alpha_init,
        "graph_microbatch_size": options.graph_microbatch_size,
        "training_mode": options.mode,
        "token_microbatch_size": options.token_microbatch_size,
        "gate_lr": options.gate_lr,
        "mixer_lr": options.mixer_lr,
        "gate_lr_scheduler": _plain_scheduler(options.gate_scheduler),
        "mixer_lr_scheduler": _plain_scheduler(options.mixer_scheduler),
        "freeze_gate": options.freeze_gate,
    }


def _validate_resume_config(saved, current) -> None:
    if saved != current:
        differing = sorted(key for key in set(saved) | set(current) if saved.get(key) != current.get(key))
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


def _materialize_context_result(result) -> dict[str, object]:
    materialized = dict(result)
    for key in ("gate_loss", "graph_loss", "joint_loss", "validation_loss"):
        value = materialized.get(key)
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
    log_metrics: bool = True,
):
    """Run one context and optionally emit its W&B metrics."""

    device = trainer.scorer.device
    timing = PhaseTiming(device)
    previous_timing = trainer.timing
    trainer.timing = timing
    try:
        if validation:
            validation_result = trainer.evaluate_context(example)
            result = {
                "gate_loss": None,
                "graph_loss": None,
                "joint_loss": None,
                "validation_loss": validation_result.loss,
                "gate_steps": 0,
                "mixer_steps": 0,
            }
        else:
            result = trainer.train_context(example, mode=mode)
            result["validation_loss"] = None
    finally:
        trainer.timing = previous_timing
    elapsed = timing.resolve()
    result = _materialize_context_result(result)
    metrics = {}
    if validation:
        metrics["validation/bce"] = result["validation_loss"]
    elif result["joint_loss"] is not None:
        metrics["train/bce"] = result["joint_loss"]
    else:
        if result["gate_loss"] is not None:
            metrics["train/gate_bce"] = result["gate_loss"]
        if result["graph_loss"] is not None:
            metrics["train/mixer_bce"] = result["graph_loss"]
    for key, value in elapsed.items():
        key = key.replace("graph_", "mixer_", 1).replace(
            "_seconds", "_seconds_per_token"
        )
        metrics[f"timing/{key}"] = value / example.sequence_length
    if not validation:
        if trainer.gate_optimizer is not None:
            metrics["train/gate_learning_rate"] = _optimizer_lr(trainer.gate_optimizer)
        if trainer.mixer_optimizer is not None:
            metrics["train/mixer_learning_rate"] = _optimizer_lr(trainer.mixer_optimizer)
    for optimizer in (trainer.gate_optimizer, trainer.mixer_optimizer):
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
    if log_metrics:
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
    microbatch = resolve_graph_microbatch_size(options.graph_microbatch_size, layers, heads)
    options = replace(options, graph_microbatch_size=microbatch)
    gates, options = _student_gates(teacher, config, options)
    scorer = ImplicitGraphScorer(
        gates,
        teacher.config,
        graph_dim=options.graph_dim,
        graph_microbatch_size=microbatch,
        gram_normalization=options.gram_normalization,
        leaky_relu_slope=options.leaky_relu_slope,
        alpha_init=options.alpha_init,
        compute_dtype=None if options.compute_dtype is None else parse_compute_dtype(options.compute_dtype),
    )
    if resume_payload is None and options.gate_checkpoint not in {None, "fastkvzip"}:
        load_gate_checkpoint(scorer, options.gate_checkpoint)
    gate_optimizer, mixer_optimizer = build_adamw_optimizers(
        scorer,
        gate_lr=options.gate_lr,
        mixer_lr=options.mixer_lr,
        weight_decay=options.weight_decay,
        gate_frozen=options.freeze_gate,
        mixer_frozen=False,
    )
    gate_scheduler = (
        build_scheduler(gate_optimizer, options.gate_scheduler)
        if gate_optimizer is not None
        else None
    )
    mixer_scheduler = (
        build_scheduler(mixer_optimizer, options.mixer_scheduler)
        if mixer_optimizer is not None
        else None
    )
    trainer = GraphTrainer(
        scorer,
        gate_optimizer=gate_optimizer,
        mixer_optimizer=mixer_optimizer,
        gate_scheduler=gate_scheduler,
        mixer_scheduler=mixer_scheduler,
        token_microbatch_size=options.token_microbatch_size,
        graph_microbatch_size=microbatch,
    )
    checkpoint_config = normalized_checkpoint_config(
        model_id=options.model_id, scorer=scorer, options=options, query_groups=query_groups
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
    resume_run_id = resume_payload.get("wandb_run_id") if resume_payload is not None else None
    run = _initialize_wandb(options, wandb_module, run_id=resume_run_id)
    succeeded = False
    try:
        _set_seed(options.seed)
        teacher = build_teacher(options.model_id, model_factory=model_factory)
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
                mixer_optimizer=trainer.mixer_optimizer,
                gate_scheduler=trainer.gate_scheduler,
                mixer_scheduler=trainer.mixer_scheduler,
            )
            cursor = copy.deepcopy(resume_payload["data_cursor"])
            training_prefix = resume_payload["prefix_ids"].detach().to("cpu").clone()
            del resume_payload
        else:
            cursor = initial_cursor()
            training_prefix = None
        if hasattr(run, "config"):
            run.config.update(
                {
                    **checkpoint_config,
                    "save_strategy": options.save_strategy,
                    "save_every": options.save_every,
                    "eval_strategy": options.eval_strategy,
                    "eval_every": options.eval_every,
                },
                allow_val_change=True,
            )
        wrappers = {}

        def make_example(key):
            nonlocal training_prefix
            cache_path = (
                None
                if options.teacher_cache_dir is None
                else _teacher_cache_path(options.teacher_cache_dir, key)
            )
            if cache_path is not None and cache_path.exists():
                example = _load_teacher_cache(
                    cache_path,
                    key=key,
                    model_id=options.model_id,
                    prefill_chunk=options.prefill_chunk,
                )
            else:
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
                if cache_path is not None:
                    _save_teacher_cache_if_missing(
                        cache_path,
                        example,
                        model_id=options.model_id,
                        prefill_chunk=options.prefill_chunk,
                    )
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
                mixer_optimizer=trainer.mixer_optimizer,
                gate_scheduler=trainer.gate_scheduler,
                mixer_scheduler=trainer.mixer_scheduler,
            )

        def evaluate():
            nonlocal cursor
            losses = []
            for key in VALIDATION_KEYS:
                example = make_example(key)
                result, _ = run_and_log_context(
                    trainer,
                    example,
                    mode=options.mode,
                    validation=True,
                    run=run,
                    step=cursor["wandb_step"],
                    log_metrics=False,
                )
                del example
                losses.append(result["validation_loss"])
            validation_mean = sum(losses) / len(losses)
            trainer.step_validation(validation_mean)
            run.log({"validation/bce": validation_mean}, step=cursor["wandb_step"])
            cursor["wandb_step"] += 1
            previous_best = cursor["best_validation_bce"]
            cursor["best_validation_bce"] = min(previous_best, validation_mean)
            if validation_mean < previous_best:
                save("best")

        processed_contexts = 0
        last_saved = True
        while cursor["epoch"] < options.epochs and (
            options.max_contexts is None or processed_contexts < options.max_contexts
        ):
            example = make_example(TRAIN_KEYS[cursor["offset"]])
            run_and_log_context(
                trainer,
                example,
                mode=options.mode,
                validation=False,
                run=run,
                step=cursor["wandb_step"],
            )
            del example
            cursor, completed_epoch = advance_train_cursor(cursor)
            processed_contexts += 1
            last_saved = False
            train_steps = training_context_steps(cursor)
            save_due = cadence_due(
                options.save_strategy,
                options.save_every,
                train_steps=train_steps,
                completed_epoch=completed_epoch,
            )
            eval_due = cadence_due(
                options.eval_strategy,
                options.eval_every,
                train_steps=train_steps,
                completed_epoch=completed_epoch,
            )
            if eval_due:
                evaluate()
            if save_due:
                save("last")
                last_saved = True
        if processed_contexts and not last_saved:
            save("last")
        succeeded = True
        return options.output_dir / "last.pt"
    finally:
        run.finish(exit_code=0 if succeeded else 1)


def main(argv=None) -> None:
    run_training(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
