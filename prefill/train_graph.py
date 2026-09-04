"""Train the streamed implicit whole-context FastKVzip mixer."""

from __future__ import annotations

import argparse
import copy
import gc
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
    ACTIVATION_ORDER,
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
from tqdm import tqdm


def _auto_or_int(value: str):
    return "auto" if value == "auto" else int(value)


def _max_or_int(value: str):
    return "max" if value == "max" else int(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("graph_checkpoints"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-contexts", type=int)
    parser.add_argument(
        "--train-context-count",
        type=int,
        help="number of regular 10K-30K FineWeb training contexts (default: 29)",
    )
    parser.add_argument("--save-strategy", choices=("epochs", "steps"), default="epochs")
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--save-best", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval-strategy", choices=("epochs", "steps"), default="epochs")
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--prefill-chunk", type=int)
    parser.add_argument("--teacher-cache-dir", type=Path)
    parser.add_argument("--teacher-cache-scores-only", action="store_true")

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
        "--subgraph-size",
        type=int,
        help="independent non-overlapping graph size (default: whole context)",
    )
    parser.add_argument(
        "--subgraphs-per-step",
        type=_max_or_int,
        help="subgraphs per optimizer update (default: max)",
    )

    parser.add_argument(
        "--training-mode",
        "--mode",
        dest="mode",
        choices=("two_phase", "two-phase", "joint"),
    )
    parser.add_argument("--gate-lr", type=float)
    parser.add_argument("--mixer-lr", "--graph-lr", dest="mixer_lr", type=float)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--adamw-eps", type=float)
    parser.add_argument("--amsgrad", action=argparse.BooleanOptionalAction, default=None)
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
    train_context_count: int
    save_strategy: str
    save_every: int
    save_best: bool
    eval_strategy: str
    eval_every: int
    seed: int
    resume: Path | None
    prefill_chunk: int
    teacher_cache_dir: Path | None
    teacher_cache_scores_only: bool
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
    subgraph_size: int | None
    subgraphs_per_step: str | int
    mode: str
    gate_lr: float
    mixer_lr: float
    weight_decay: float
    adamw_eps: float
    amsgrad: bool
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
    if resume_payload:
        saved.setdefault("adamw_eps", 1e-8)
        saved.setdefault("amsgrad", False)
        if "subgraph_size" in saved:
            saved.setdefault("subgraphs_per_step", "max")
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
    train_context_count = _positive_int(
        "train-context-count",
        _pick(args.train_context_count, saved, "train_context_count", 29),
    )
    save_every = _positive_int("save-every", args.save_every)
    eval_every = _positive_int("eval-every", args.eval_every)
    weight_decay = _positive_finite("weight decay", args.weight_decay, allow_zero=True)
    adamw_eps = _positive_finite(
        "AdamW epsilon", _pick(args.adamw_eps, saved, "adamw_eps", 1e-8)
    )
    amsgrad = bool(_pick(args.amsgrad, saved, "amsgrad", False))

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
    saved_subgraph_size = saved.get("subgraph_size")
    if (
        resume_payload
        and args.subgraph_size is not None
        and args.subgraph_size != saved_subgraph_size
    ):
        raise ValueError("resume configuration conflicts for subgraph_size")
    subgraph_size = saved_subgraph_size if resume_payload else args.subgraph_size
    subgraphs_per_step = _pick(
        args.subgraphs_per_step, saved, "subgraphs_per_step", "max"
    )
    if subgraph_size is not None:
        _positive_int("subgraph-size", subgraph_size)
        if mode != "joint":
            raise ValueError("--subgraph-size requires joint training")
        if token_microbatch_size % subgraph_size:
            raise ValueError("--subgraph-size must divide --token-microbatch-size")
        if subgraphs_per_step != "max":
            _positive_int("subgraphs-per-step", subgraphs_per_step)
            capacity = token_microbatch_size // subgraph_size
            if subgraphs_per_step % capacity:
                raise ValueError(
                    "--subgraphs-per-step must be divisible by the number of "
                    "subgraphs in a token microbatch"
                )
    elif args.subgraphs_per_step is not None or subgraphs_per_step != "max":
        raise ValueError("--subgraphs-per-step requires --subgraph-size")
    gate_scheduler = _scheduler_option(args, "gate", saved)
    mixer_scheduler = _scheduler_option(args, "mixer", saved)
    if (
        mode == "two_phase"
        and not freeze_gate
        and gate_scheduler is not None
        and gate_scheduler.name == "LinearWarmupCosineLR"
    ):
        raise ValueError("LinearWarmupCosineLR requires joint training for a trainable gate")
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
    if args.teacher_cache_scores_only and args.teacher_cache_dir is None:
        raise ValueError("--teacher-cache-scores-only requires --teacher-cache-dir")

    return TrainingOptions(
        model_id=args.model,
        output_dir=args.output_dir,
        epochs=args.epochs,
        max_contexts=args.max_contexts,
        train_context_count=train_context_count,
        save_strategy=args.save_strategy,
        save_every=save_every,
        save_best=args.save_best,
        eval_strategy=args.eval_strategy,
        eval_every=eval_every,
        seed=args.seed,
        resume=args.resume,
        prefill_chunk=prefill_chunk,
        teacher_cache_dir=args.teacher_cache_dir,
        teacher_cache_scores_only=args.teacher_cache_scores_only,
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
        subgraph_size=subgraph_size,
        subgraphs_per_step=subgraphs_per_step,
        mode=mode,
        gate_lr=gate_lr,
        mixer_lr=mixer_lr,
        weight_decay=weight_decay,
        adamw_eps=adamw_eps,
        amsgrad=amsgrad,
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


def _validated_teacher_scores(
    teacher_scores, sequence_length: int, *, cached: bool
) -> torch.Tensor:
    label = "cached teacher scores" if cached else "teacher scores"
    if (
        not isinstance(teacher_scores, torch.Tensor)
        or teacher_scores.ndim != 4
        or teacher_scores.size(1) != 1
    ):
        raise ValueError(f"{label} must have shape [layers,1,heads,tokens]")
    if teacher_scores.size(-1) != sequence_length:
        raise ValueError(f"{label} length does not match context")
    return teacher_scores


def _teacher_context_payload(
    kv, dataset_name: str, dataset_index: int, *, scores: bool
):
    start_idx, end_idx = int(kv.start_idx), int(kv.end_idx)
    sequence_length = end_idx - start_idx
    payload = {
        "dataset_name": dataset_name,
        "dataset_index": dataset_index,
        "token_ids": _normal_capture(kv.prefill_ids[:, start_idx:end_idx]),
        "prefix_ids": _normal_capture(kv.prefill_ids[:, :start_idx]),
        "sequence_length": sequence_length,
    }
    if scores:
        teacher_scores = torch.stack(kv.score, dim=0) if isinstance(
            kv.score, (list, tuple)
        ) else kv.score
        if (
            isinstance(teacher_scores, torch.Tensor)
            and teacher_scores.ndim == 4
            and teacher_scores.size(-1) == end_idx
        ):
            teacher_scores = teacher_scores[..., start_idx:end_idx]
        payload["teacher_scores"] = _normal_capture(
            _validated_teacher_scores(
                teacher_scores, sequence_length, cached=False
            )
        )
    return payload


def _hidden_from_kv(kv) -> list[torch.Tensor]:
    hidden = []
    for layer_hidden in kv.hidden_cache:
        if layer_hidden.ndim != 3 or layer_hidden.size(0) != 1:
            raise ValueError("cached hidden states must have shape [1,prefix+tokens,dim]")
        normal = _normal_capture(layer_hidden)
        hidden.append(normal[:, int(kv.start_idx):int(kv.end_idx), :])
    return hidden


def _teacher_example_from_payload(payload, hidden_by_layer) -> TeacherExample:
    sequence_length = payload["sequence_length"]
    hidden = tuple(hidden_by_layer)
    scores = _validated_teacher_scores(
        payload["teacher_scores"], sequence_length, cached=True
    )
    if not hidden:
        raise ValueError("hidden layer replay is empty")
    if any(
        not isinstance(tensor, torch.Tensor)
        or tensor.ndim not in {2, 3}
        or tensor.size(-2) != sequence_length
        or (tensor.ndim == 3 and tensor.size(0) != 1)
        for tensor in hidden
    ):
        raise ValueError("hidden layer replay does not match the cached sequence length")
    if scores.size(0) != len(hidden):
        raise ValueError("cached teacher scores do not match hidden layer replay")
    return TeacherExample.from_owned_cpu(
        dataset_name=payload["dataset_name"],
        dataset_index=payload["dataset_index"],
        token_ids=payload["token_ids"],
        hidden_by_layer=hidden,
        teacher_scores=scores,
        prefix_ids=payload["prefix_ids"],
        sequence_length=sequence_length,
    )


def teacher_example_from_kv(kv, dataset_name: str, dataset_index: int) -> TeacherExample:
    """Transfer retained normal CPU hidden tensors to a teacher example without cloning."""

    payload = _teacher_context_payload(kv, dataset_name, dataset_index, scores=True)
    return _teacher_example_from_payload(payload, _hidden_from_kv(kv))


def _teacher_example_from_cached_scores(cached, kv) -> TeacherExample:
    fresh = _teacher_context_payload(
        kv, cached["dataset_name"], cached["dataset_index"], scores=False
    )
    for field, label in (
        ("sequence_length", "sequence length"),
        ("token_ids", "token IDs"),
        ("prefix_ids", "prefix IDs"),
    ):
        mismatch = (
            fresh[field] != cached[field]
            if field == "sequence_length"
            else not torch.equal(fresh[field], cached[field])
        )
        if mismatch:
            raise ValueError(f"cached {label} do not match hidden replay")
    fresh["teacher_scores"] = cached["teacher_scores"]
    return _teacher_example_from_payload(fresh, _hidden_from_kv(kv))


def _teacher_cache_path(cache_dir: Path, key: tuple[str, int]) -> Path:
    dataset_name, dataset_index = key
    if Path(dataset_name).name != dataset_name:
        raise ValueError("teacher cache dataset name is not a safe path component")
    return cache_dir / dataset_name / f"source-{dataset_index}.pt"


def _teacher_cache_complete(
    cache_dir: Path | None,
    train_keys,
    validation_keys,
) -> bool:
    return cache_dir is not None and all(
        _teacher_cache_path(cache_dir, key).is_file()
        for key in (*train_keys, *validation_keys)
    )


def _teacher_cache_payload(
    example_or_payload,
    *,
    model_id: str,
    prefill_chunk: int,
    scores_only: bool = False,
) -> dict[str, object]:
    if isinstance(example_or_payload, TeacherExample):
        payload = {
            "dataset_name": example_or_payload.dataset_name,
            "dataset_index": example_or_payload.dataset_index,
            "token_ids": example_or_payload.token_ids,
            "hidden_by_layer": list(example_or_payload.hidden_by_layer),
            "teacher_scores": example_or_payload.teacher_scores,
            "prefix_ids": example_or_payload.prefix_ids,
            "sequence_length": example_or_payload.sequence_length,
        }
    else:
        payload = dict(example_or_payload)
        payload["hidden_by_layer"] = None
    payload["hidden_by_layer"] = None if scores_only else payload["hidden_by_layer"]
    return {
        **payload,
        "model_id": model_id,
        "prefill_chunk": prefill_chunk,
    }


def _load_teacher_cache_payload(
    path: Path,
    *,
    key: tuple[str, int],
    model_id: str,
    prefill_chunk: int,
    scores_only: bool,
) -> Mapping[str, object]:
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
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, Mapping) or set(payload) != required:
            raise ValueError("cache payload fields do not match")
        if (payload["dataset_name"], payload["dataset_index"]) != key:
            raise ValueError("cache context identity does not match")
        if payload["model_id"] != model_id or payload["prefill_chunk"] != prefill_chunk:
            raise ValueError("cache model or prefill settings do not match")
        if payload["hidden_by_layer"] is None and not scores_only:
            raise ValueError("scores-only cache requires --teacher-cache-scores-only")
        return payload
    except Exception as error:
        raise ValueError(f"invalid or incompatible teacher cache {path}: {error}") from error


def _load_teacher_cache(
    path: Path,
    *,
    key: tuple[str, int],
    model_id: str,
    prefill_chunk: int,
) -> TeacherExample:
    try:
        payload = _load_teacher_cache_payload(
            path,
            key=key,
            model_id=model_id,
            prefill_chunk=prefill_chunk,
            scores_only=False,
        )
        return _teacher_example_from_payload(payload, payload["hidden_by_layer"])
    except Exception as error:
        if str(error).startswith("invalid or incompatible teacher cache"):
            raise
        raise ValueError(f"invalid or incompatible teacher cache {path}: {error}") from error


def _save_teacher_cache_if_missing(
    path: Path,
    example_or_payload,
    *,
    model_id: str,
    prefill_chunk: int,
    scores_only: bool = False,
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
            _teacher_cache_payload(
                example_or_payload,
                model_id=model_id,
                prefill_chunk=prefill_chunk,
                scores_only=scores_only,
            ),
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
        "training_tokens": 0,
        "best_validation_bce": float("inf"),
        "wandb_step": 0,
    }


def advance_train_cursor(
    cursor, *, token_count: int = 0, contexts_per_epoch: int
):
    """Advance one whole training context and report epoch completion."""

    cursor = copy.deepcopy(cursor)
    if cursor.get("phase") != "train":
        raise ValueError("checkpoint cursor must be at a training context")
    cursor["training_tokens"] = int(cursor.get("training_tokens", 0)) + token_count
    cursor["offset"] += 1
    cursor["wandb_step"] += 1
    completed_epoch = cursor["offset"] == contexts_per_epoch
    if completed_epoch:
        cursor["offset"] = 0
        cursor["epoch"] += 1
    return cursor, completed_epoch


def training_context_steps(cursor, contexts_per_epoch: int) -> int:
    return int(cursor["epoch"]) * contexts_per_epoch + int(cursor["offset"])


def cadence_due(
    strategy: str,
    every: int,
    *,
    train_steps: int,
    completed_epoch: bool,
    contexts_per_epoch: int,
) -> bool:
    if strategy == "steps":
        return train_steps % every == 0
    return completed_epoch and (train_steps // contexts_per_epoch) % every == 0


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
    config = {
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
        "activation_order": ACTIVATION_ORDER,
        "alpha_init": options.alpha_init,
        "graph_microbatch_size": options.graph_microbatch_size,
        "training_mode": options.mode,
        "token_microbatch_size": options.token_microbatch_size,
        "gate_lr": options.gate_lr,
        "mixer_lr": options.mixer_lr,
        "adamw_eps": options.adamw_eps,
        "amsgrad": options.amsgrad,
        "gate_lr_scheduler": _plain_scheduler(options.gate_scheduler),
        "mixer_lr_scheduler": _plain_scheduler(options.mixer_scheduler),
        "freeze_gate": options.freeze_gate,
        "train_context_count": options.train_context_count,
    }
    if options.subgraph_size is not None:
        config["subgraph_size"] = options.subgraph_size
        config["subgraphs_per_step"] = options.subgraphs_per_step
    return config


def _validate_resume_config(saved, current) -> None:
    if saved != current:
        differing = sorted(key for key in set(saved) | set(current) if saved.get(key) != current.get(key))
        raise ValueError(f"resume configuration conflicts for: {', '.join(differing)}")


def _persistent_wandb_run_id(options: TrainingOptions, module, checkpoint_run_id=None):
    if options.wandb_mode != "online":
        return checkpoint_run_id
    path = options.output_dir / "wandb_run_id.txt"
    if path.exists():
        run_id = path.read_text(encoding="utf-8").strip()
        if not run_id:
            raise ValueError(f"W&B run ID file is empty: {path}")
        if checkpoint_run_id is not None and run_id != checkpoint_run_id:
            raise ValueError("W&B run ID conflicts with the resume checkpoint")
        return run_id
    run_id = checkpoint_run_id or module.util.generate_id()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as temporary:
        temporary.write(f"{run_id}\n")
        temporary_path = Path(temporary.name)
    try:
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return run_id


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
    for key in (
        "gate_loss",
        "graph_loss",
        "joint_loss",
        "validation_loss",
        "gradient_norm",
        "gate_gradient_norm",
        "mixer_gradient_norm",
    ):
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
    fractional_epoch: float | None = None,
    cumulative_training_tokens: int | None = None,
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
        metrics["train/grad_norm"] = result["gradient_norm"]
        metrics["train/gate_grad_norm"] = result["gate_gradient_norm"]
        metrics["train/mixer_grad_norm"] = result["mixer_gradient_norm"]
        alpha = trainer.scorer.mixer.alpha.detach().float()
        metrics["train/mean_alpha"] = float(alpha.mean().item())
        if fractional_epoch is not None:
            metrics["train/epoch"] = fractional_epoch
        if cumulative_training_tokens is not None:
            metrics["train/tokens"] = cumulative_training_tokens
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


def _make_components(teacher, options, resume_payload, *, total_steps):
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
        eps=options.adamw_eps,
        amsgrad=options.amsgrad,
        gate_frozen=options.freeze_gate,
        mixer_frozen=False,
    )
    gate_scheduler = (
        build_scheduler(gate_optimizer, options.gate_scheduler, total_steps=total_steps)
        if gate_optimizer is not None
        else None
    )
    mixer_scheduler = (
        build_scheduler(mixer_optimizer, options.mixer_scheduler, total_steps=total_steps)
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
        subgraph_size=options.subgraph_size,
        subgraphs_per_step=options.subgraphs_per_step,
    )
    checkpoint_config = normalized_checkpoint_config(
        model_id=options.model_id, scorer=scorer, options=options, query_groups=query_groups
    )
    return options, scorer, trainer, checkpoint_config


def run_training(
    args,
    *,
    model_factory=None,
    data_builder=None,
    wrapper_factory=None,
    wandb_module=wandb,
    progress_factory=tqdm,
):
    resume_payload = _load_payload(args.resume)
    gate_payload = None
    if args.gate_checkpoint not in {None, "fastkvzip"}:
        gate_payload = _load_payload(args.gate_checkpoint)
    options = resolve_options(args, resume_payload, gate_payload)
    del gate_payload
    resume_run_id = resume_payload.get("wandb_run_id") if resume_payload is not None else None
    resume_run_id = _persistent_wandb_run_id(
        options, wandb_module, checkpoint_run_id=resume_run_id
    )
    run = _initialize_wandb(options, wandb_module, run_id=resume_run_id)
    succeeded = False
    progress = None
    try:
        _set_seed(options.seed)
        teacher = build_teacher(options.model_id, model_factory=model_factory)
        _, layers, heads, _ = _model_dimensions(teacher)
        resolve_graph_microbatch_size(options.graph_microbatch_size, layers, heads)
        if data_builder is None or wrapper_factory is None:
            from data import DataWrapper, load_fineweb_training

            data_builder = data_builder or load_fineweb_training
            wrapper_factory = wrapper_factory or DataWrapper
        datasets, train_keys, validation_keys = data_builder(
            options.train_context_count
        )
        contexts_per_epoch = len(train_keys)
        options, scorer, trainer, checkpoint_config = _make_components(
            teacher, options, resume_payload,
            total_steps=options.epochs * contexts_per_epoch,
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
            cursor.setdefault("training_tokens", 0)
            training_prefix = resume_payload["prefix_ids"].detach().to("cpu").clone()
            del resume_payload
        else:
            cursor = initial_cursor()
            training_prefix = None
        wrappers = {}

        def unload_teacher_if_cached():
            nonlocal teacher
            if (
                not options.teacher_cache_scores_only
                and teacher is not None
                and _teacher_cache_complete(
                    options.teacher_cache_dir, train_keys, validation_keys
                )
            ):
                wrappers.clear()
                teacher = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print("Teacher cache complete; unloaded base model")

        unload_teacher_if_cached()
        initial_train_steps = training_context_steps(cursor, contexts_per_epoch)
        progress_total = options.epochs * contexts_per_epoch
        if options.max_contexts is not None:
            progress_total = min(progress_total, initial_train_steps + options.max_contexts)
        progress = progress_factory(
            total=progress_total,
            initial=initial_train_steps,
            desc="Training",
            unit="context",
            position=1,
        )
        if hasattr(run, "config"):
            run.config.update(
                {
                    **checkpoint_config,
                    "save_strategy": options.save_strategy,
                    "save_every": options.save_every,
                    "save_best": options.save_best,
                    "eval_strategy": options.eval_strategy,
                    "eval_every": options.eval_every,
                },
                allow_val_change=True,
            )
        def ensure_teacher():
            nonlocal teacher
            if teacher is None:
                teacher = build_teacher(options.model_id, model_factory=model_factory)
            return teacher

        def make_example(key):
            nonlocal training_prefix
            unload_teacher_if_cached()
            cache_path = (
                None
                if options.teacher_cache_dir is None
                else _teacher_cache_path(options.teacher_cache_dir, key)
            )
            if (
                cache_path is not None
                and cache_path.exists()
                and not options.teacher_cache_scores_only
            ):
                example = _load_teacher_cache(
                    cache_path,
                    key=key,
                    model_id=options.model_id,
                    prefill_chunk=options.prefill_chunk,
                )
            else:
                active_teacher = ensure_teacher()
                dataset_name, dataset_index = key
                if dataset_name not in wrappers:
                    wrappers[dataset_name] = wrapper_factory(
                        dataset_name,
                        datasets[dataset_name],
                        active_teacher,
                    )
                if training_prefix is not None:
                    active_teacher.sys_prompt_ids = training_prefix.to(
                        active_teacher.device
                    )
                if options.teacher_cache_scores_only:
                    if cache_path is not None and cache_path.exists():
                        score_payload = _load_teacher_cache_payload(
                            cache_path,
                            key=key,
                            model_id=options.model_id,
                            prefill_chunk=options.prefill_chunk,
                            scores_only=True,
                        )
                    else:
                        kv = wrappers[dataset_name].prefill_context(
                            dataset_index,
                            prefill_chunk=options.prefill_chunk,
                            save_hidden=False,
                            do_score=True,
                        )
                        score_payload = _teacher_context_payload(
                            kv, dataset_name, dataset_index, scores=True
                        )
                        if cache_path is not None:
                            _save_teacher_cache_if_missing(
                                cache_path,
                                score_payload,
                                model_id=options.model_id,
                                prefill_chunk=options.prefill_chunk,
                                scores_only=True,
                            )
                        del kv
                    kv = wrappers[dataset_name].prefill_context(
                        dataset_index,
                        prefill_chunk=options.prefill_chunk,
                        save_hidden=True,
                        do_score=False,
                    )
                    example = _teacher_example_from_cached_scores(score_payload, kv)
                    del kv
                else:
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
            for key in validation_keys:
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
            if options.save_best and validation_mean < previous_best:
                save("best")

        processed_contexts = 0
        last_saved = True
        while cursor["epoch"] < options.epochs and (
            options.max_contexts is None or processed_contexts < options.max_contexts
        ):
            example = make_example(train_keys[cursor["offset"]])
            next_cursor, completed_epoch = advance_train_cursor(
                cursor,
                token_count=example.sequence_length,
                contexts_per_epoch=contexts_per_epoch,
            )
            _, metrics = run_and_log_context(
                trainer,
                example,
                mode=options.mode,
                validation=False,
                run=run,
                step=cursor["wandb_step"],
                fractional_epoch=training_context_steps(next_cursor, contexts_per_epoch)
                / contexts_per_epoch,
                cumulative_training_tokens=next_cursor["training_tokens"],
            )
            del example
            cursor = next_cursor
            processed_contexts += 1
            last_saved = False
            progress.update(1)
            progress.set_postfix(
                {
                    key.removeprefix("train/"): value
                    for key, value in metrics.items()
                    if key.startswith("train/")
                }
            )
            train_steps = training_context_steps(cursor, contexts_per_epoch)
            save_due = cadence_due(
                options.save_strategy,
                options.save_every,
                train_steps=train_steps,
                completed_epoch=completed_epoch,
                contexts_per_epoch=contexts_per_epoch,
            )
            eval_due = cadence_due(
                options.eval_strategy,
                options.eval_every,
                train_steps=train_steps,
                completed_epoch=completed_epoch,
                contexts_per_epoch=contexts_per_epoch,
            )
            if eval_due:
                progress.set_description("Validating")
                evaluate()
                progress.set_description("Training")
            if save_due:
                save("last")
                last_saved = True
        if processed_contexts and not last_saved:
            save("last")
        succeeded = True
        return options.output_dir / "last.pt"
    finally:
        if progress is not None:
            progress.close()
        run.finish(exit_code=0 if succeeded else 1)


def main(argv=None) -> None:
    run_training(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
