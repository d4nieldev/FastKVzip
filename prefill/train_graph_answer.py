"""Train the graph scorer from answer-only causal cross-entropy."""

from __future__ import annotations

import argparse
import copy
import math
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

import torch
import wandb
from graph import (
    ImplicitGraphScorer,
    answer_objective,
    build_adamw_optimizers,
    build_scheduler,
    compact_context_kv,
    fallback_validation_split,
    freeze_llm,
    load_checkpoint,
    parse_compute_dtype,
    replay_score_gradients,
    retention_ratio,
    save_checkpoint,
    score_context_subgraphs,
    validate_answer_training_model_identity,
)
from graph.training import _model_gradient_norms
from tqdm import tqdm

import train_graph


OBJECTIVE = "answer-only-causal-ce-v1"
DEDUPLICATION_VERSION = "exact-context-question-v1"
ANSWER_GENERATION_PROTOCOL = "runtime-teacher-full-unpruned-prefill-v1"
ANSWER_CACHE_IDENTITY_PROTOCOL = "agentic-content-model-template-generation-v1"

TRAIN_LOG_KEYS = frozenset(
    {
        "train/answer_nll",
        "train/answer_token_accuracy",
        "train/grad_norm",
        "train/gate_grad_norm",
        "train/mixer_grad_norm",
        "train/score_grad_norm",
        "train/retained_score_grad_norm",
        "train/evicted_score_grad_norm",
        "train/mean_alpha",
        "train/gate_learning_rate",
        "train/mixer_learning_rate",
        "train/epoch",
        "train/tokens",
        "train/examples",
        "train/optimizer_step",
        "train/batch_examples",
    }
)
VALIDATION_LOG_KEYS = frozenset(
    {"validation/answer_nll", "validation/answer_token_accuracy"}
)


class _StoreExplicit(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        setattr(namespace, f"_{self.dest}_explicit", True)


class _StoreExplicitBoolean(argparse.BooleanOptionalAction):
    def __call__(self, parser, namespace, values, option_string=None):
        super().__call__(parser, namespace, values, option_string)
        setattr(namespace, f"_{self.dest}_explicit", True)


def _auto_or_int(value: str):
    return "auto" if value == "auto" else int(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--resume", type=Path)
    source.add_argument("--graph-checkpoint", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--output-dir", type=Path, default=Path("graph_answer_checkpoints"))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--data", default="agentic", action=_StoreExplicit)
    parser.add_argument("--train-context-start", type=int)
    parser.add_argument(
        "--train-context-count", type=int,
        help="question-context pairs to select before the validation split; reused each epoch",
    )
    parser.add_argument("--answer-cache-dir", type=Path)
    parser.add_argument("--prefill-chunk", type=int)
    parser.add_argument(
        "--gradient-accumulation-steps", type=int,
        help="questions per optimizer update (default: 1); average questions equally",
    )
    parser.add_argument(
        "--shuffle-data", action=_StoreExplicitBoolean, default=True,
        help="shuffle training question-context pairs each epoch (default: enabled for new runs)",
    )

    parser.add_argument(
        "--retention-scheduler",
        choices=("uniform", "linear"),
        default="linear",
        action=_StoreExplicit,
        help=(
            "uniform samples once per example; linear decays from --retention-max "
            "to --retention-min over the global optimizer-step horizon and never "
            "resets at epoch boundaries"
        ),
    )
    parser.add_argument(
        "--retention-min",
        type=float,
        default=0.10,
        action=_StoreExplicit,
        help="lower bound for uniform; final value for linear (default: 0.10)",
    )
    parser.add_argument(
        "--retention-max",
        type=float,
        default=0.30,
        action=_StoreExplicit,
        help="upper bound for uniform; initial value for linear (default: 0.30)",
    )
    parser.add_argument("--validation-retention-ratio", type=float, required=True)
    parser.add_argument(
        "--ste-temperature", type=float, default=1.0, action=_StoreExplicit
    )

    parser.add_argument("--gate-dim", type=int)
    parser.add_argument("--gate-sink", type=int)
    parser.add_argument("--graph-dim", type=int)
    parser.add_argument("--gram-normalization", choices=("token-count", "none"))
    parser.add_argument("--leaky-relu-slope", type=float)
    parser.add_argument("--alpha-init", type=float)
    parser.add_argument("--graph-microbatch-size", type=_auto_or_int)
    parser.add_argument("--token-microbatch-size", type=int)
    parser.add_argument("--subgraph-size", type=int)

    parser.add_argument("--gate-lr", type=float)
    parser.add_argument("--mixer-lr", "--graph-lr", dest="mixer_lr", type=float)
    parser.add_argument("--weight-decay", type=float)
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
        "--save-strategy",
        choices=("epochs", "steps"),
        default="epochs",
        action=_StoreExplicit,
    )
    parser.add_argument("--save-every", type=int, default=1, action=_StoreExplicit)
    parser.add_argument("--save-best", action=_StoreExplicitBoolean, default=True)
    parser.add_argument(
        "--eval-strategy",
        choices=("epochs", "steps"),
        default="epochs",
        action=_StoreExplicit,
    )
    parser.add_argument("--eval-every", type=int, default=1, action=_StoreExplicit)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled"), default="online"
    )
    parser.add_argument("--wandb-project", default="answer-graph-fastkvzip")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-name")
    return parser


@dataclass(frozen=True)
class AnswerTrainingOptions:
    initialization: str
    model_id: str
    output_dir: Path
    epochs: int
    gradient_accumulation_steps: int
    shuffle_data: bool
    data: str
    train_context_start: int
    train_context_count: int
    answer_cache_dir: Path | None
    prefill_chunk: int
    retention_scheduler: str
    retention_min: float
    retention_max: float
    validation_retention_ratio: float
    ste_temperature: float
    gate_dim: int
    gate_sink: int
    compute_dtype: str | None
    graph_dim: int
    gram_normalization: str
    leaky_relu_slope: float
    alpha_init: float
    graph_microbatch_size: str | int
    token_microbatch_size: int
    subgraph_size: int | None
    gate_lr: float
    mixer_lr: float
    weight_decay: float
    adamw_eps: float
    amsgrad: bool
    gate_scheduler: object | None
    mixer_scheduler: object | None
    save_strategy: str
    save_every: int
    save_best: bool
    eval_strategy: str
    eval_every: int
    seed: int
    wandb_mode: str
    wandb_project: str
    wandb_entity: str | None
    wandb_name: str | None

    @property
    def mode(self) -> str:
        return "joint"

    @property
    def freeze_gate(self) -> bool:
        return False

    @property
    def subgraphs_per_step(self) -> str:
        return "max"

    @property
    def shuffle_subgraphs(self) -> bool:
        return False


def _non_negative_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


_positive_int = train_graph._positive_int


def _finite(name: str, value: float, *, positive=True, allow_zero=False) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or (positive and (value < 0 or (value == 0 and not allow_zero)))
    ):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return float(value)


def _payload_config(payload) -> Mapping[str, object]:
    if payload is None:
        return {}
    config = payload.get("config") if isinstance(payload, Mapping) else None
    if not isinstance(config, Mapping):
        raise ValueError("checkpoint config must be a mapping")
    return config


def normalized_answer_resume_config(config):
    """Supply historical defaults without modifying an existing checkpoint."""
    config = dict(config)
    config.setdefault("gradient_accumulation_steps", 1)
    config.setdefault("shuffle_data", False)
    return config


def _pick(
    args,
    name: str,
    saved: Mapping[str, object],
    default,
    *,
    strict: bool,
    saved_name: str | None = None,
):
    saved_name = saved_name or name
    cli_value = getattr(args, name)
    if cli_value is None:
        return saved.get(saved_name, default)
    if strict and saved_name in saved and cli_value != saved[saved_name]:
        raise ValueError(f"checkpoint configuration conflicts for {saved_name}")
    return cli_value


def _explicit_pick(
    args, name: str, saved, default, *, strict: bool, saved_name: str | None = None
):
    saved_name = saved_name or name
    explicit = bool(getattr(args, f"_{name}_explicit", False))
    cli_value = getattr(args, name)
    if strict and saved_name in saved:
        if explicit and cli_value != saved[saved_name]:
            raise ValueError(f"checkpoint configuration conflicts for {saved_name}")
        return saved[saved_name]
    return cli_value


def _ratio(name: str, value: float) -> float:
    value = _finite(name, value)
    if value > 1:
        raise ValueError(f"{name} must be at most 1")
    return value


def resolve_options(args, checkpoint_payload=None) -> AnswerTrainingOptions:
    """Resolve checkpoint-selected identity and validate before expensive work."""

    initialization = (
        "resume"
        if args.resume is not None
        else "graph-checkpoint"
        if args.graph_checkpoint is not None
        else "fresh"
    )
    if initialization != "fresh" and checkpoint_payload is None:
        raise ValueError("checkpoint payload is required")
    saved = _payload_config(checkpoint_payload)
    strict_resume = initialization == "resume"
    runtime_saved = normalized_answer_resume_config(saved) if strict_resume else {}
    if strict_resume and saved.get("objective") != OBJECTIVE:
        raise ValueError("resume checkpoint is not answer-supervised")
    checkpoint_model = (
        checkpoint_payload.get("model_id") if checkpoint_payload is not None else None
    )
    if checkpoint_model is not None and saved.get("model_id") != checkpoint_model:
        raise ValueError("checkpoint model identifiers disagree")
    if checkpoint_model is not None:
        if args.model is not None and args.model != checkpoint_model:
            raise ValueError("--model must exactly match the checkpoint model_id")
        model_id = checkpoint_model
    else:
        model_id = args.model
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("--model is required when no checkpoint selects it")
    validate_answer_training_model_identity(model_id)

    data = _explicit_pick(
        args,
        "data",
        saved,
        "agentic",
        strict=strict_resume,
        saved_name="dataset",
    )
    epochs = _positive_int(
        "epochs", _pick(args, "epochs", runtime_saved, 1, strict=strict_resume)
    )
    gradient_accumulation_steps = _positive_int(
        "gradient-accumulation-steps",
        _pick(args, "gradient_accumulation_steps", runtime_saved, 1, strict=strict_resume),
    )
    shuffle_data = bool(
        _explicit_pick(args, "shuffle_data", runtime_saved, True, strict=strict_resume)
    )
    train_context_start = _non_negative_int(
        "train-context-start",
        _pick(
            args,
            "train_context_start",
            runtime_saved,
            0,
            strict=strict_resume,
        ),
    )
    train_context_count = _positive_int(
        "train-context-count",
        _pick(
            args,
            "train_context_count",
            runtime_saved,
            29,
            strict=strict_resume,
        ),
    )
    retention_scheduler = _explicit_pick(
        args, "retention_scheduler", saved, "linear", strict=strict_resume
    )
    retention_min = _ratio(
        "retention-min",
        _explicit_pick(args, "retention_min", saved, 0.10, strict=strict_resume),
    )
    retention_max = _ratio(
        "retention-max",
        _explicit_pick(args, "retention_max", saved, 0.30, strict=strict_resume),
    )
    if retention_min > retention_max:
        raise ValueError("retention-min must not exceed retention-max")
    validation_ratio = _ratio(
        "validation-retention-ratio", args.validation_retention_ratio
    )
    if strict_resume and validation_ratio != saved.get("validation_retention_ratio"):
        raise ValueError(
            "checkpoint configuration conflicts for validation_retention_ratio"
        )
    ste_temperature = _finite(
        "ste-temperature",
        _explicit_pick(args, "ste_temperature", saved, 1.0, strict=strict_resume),
    )

    strict_architecture = initialization != "fresh"
    gate_dim = _positive_int(
        "gate-dim",
        int(_pick(args, "gate_dim", saved, 16, strict=strict_architecture)),
    )
    gate_sink = _positive_int(
        "gate-sink",
        int(_pick(args, "gate_sink", saved, 16, strict=strict_architecture)),
    )
    graph_dim = _positive_int(
        "graph-dim",
        int(_pick(args, "graph_dim", saved, 32, strict=strict_architecture)),
    )
    gram_normalization = _pick(
        args,
        "gram_normalization",
        saved,
        "token-count",
        strict=strict_architecture,
    )
    leaky_relu_slope = _finite(
        "leaky ReLU slope",
        _pick(
            args,
            "leaky_relu_slope",
            saved,
            0.01,
            strict=strict_architecture,
        ),
        allow_zero=True,
    )
    alpha_init = _pick(
        args, "alpha_init", saved, 0.1, strict=strict_architecture
    )
    if (
        isinstance(alpha_init, bool)
        or not isinstance(alpha_init, (int, float))
        or not math.isfinite(alpha_init)
    ):
        raise ValueError("alpha-init must be finite")
    compute_dtype = saved.get("compute_dtype") if checkpoint_payload is not None else None
    if compute_dtype is not None:
        parse_compute_dtype(compute_dtype)

    performance_saved = runtime_saved
    graph_microbatch_size = _pick(
        args,
        "graph_microbatch_size",
        performance_saved,
        "auto",
        strict=strict_resume,
    )
    token_microbatch_size = _positive_int(
        "token-microbatch-size",
        int(
            _pick(
                args,
                "token_microbatch_size",
                performance_saved,
                1000,
                strict=strict_resume,
            )
        ),
    )
    saved_subgraph = saved.get("subgraph_size")
    if strict_architecture:
        if args.subgraph_size is not None and args.subgraph_size != saved_subgraph:
            raise ValueError("checkpoint configuration conflicts for subgraph_size")
        subgraph_size = saved_subgraph
    else:
        subgraph_size = args.subgraph_size
    if subgraph_size is not None:
        subgraph_size = _positive_int("subgraph-size", int(subgraph_size))
        if token_microbatch_size % subgraph_size:
            raise ValueError("subgraph-size must divide token-microbatch-size")

    optimization_saved = saved if strict_resume else {}
    gate_lr = train_graph._positive_finite(
        "gate learning rate",
        _pick(args, "gate_lr", optimization_saved, 1e-4, strict=strict_resume),
    )
    mixer_lr = train_graph._positive_finite(
        "mixer learning rate",
        _pick(args, "mixer_lr", optimization_saved, 1e-3, strict=strict_resume),
    )
    weight_decay = train_graph._positive_finite(
        "weight decay",
        _pick(
            args,
            "weight_decay",
            optimization_saved,
            0.01,
            strict=strict_resume,
        ),
        allow_zero=True,
    )
    adamw_eps = train_graph._positive_finite(
        "AdamW epsilon",
        _pick(args, "adamw_eps", optimization_saved, 1e-8, strict=strict_resume),
    )
    amsgrad = bool(
        _pick(args, "amsgrad", optimization_saved, False, strict=strict_resume)
    )
    gate_scheduler = train_graph._scheduler_option(args, "gate", optimization_saved)
    mixer_scheduler = train_graph._scheduler_option(args, "mixer", optimization_saved)
    prefill_chunk = _positive_int(
        "prefill-chunk",
        int(
            _pick(
                args,
                "prefill_chunk",
                {"prefill_chunk": checkpoint_payload["prefill_chunk"]}
                if strict_resume
                else {},
                16000,
                strict=strict_resume,
            )
        ),
    )
    save_strategy = _explicit_pick(
        args, "save_strategy", runtime_saved, "epochs", strict=strict_resume
    )
    save_every = _positive_int(
        "save-every",
        _explicit_pick(args, "save_every", runtime_saved, 1, strict=strict_resume),
    )
    save_best = bool(
        _explicit_pick(args, "save_best", runtime_saved, True, strict=strict_resume)
    )
    eval_strategy = _explicit_pick(
        args, "eval_strategy", runtime_saved, "epochs", strict=strict_resume
    )
    eval_every = _positive_int(
        "eval-every",
        _explicit_pick(args, "eval_every", runtime_saved, 1, strict=strict_resume),
    )
    seed = _non_negative_int(
        "seed", _pick(args, "seed", runtime_saved, 0, strict=strict_resume)
    )

    return AnswerTrainingOptions(
        initialization=initialization,
        model_id=model_id,
        output_dir=args.output_dir,
        epochs=epochs,
        gradient_accumulation_steps=gradient_accumulation_steps,
        shuffle_data=shuffle_data,
        data=str(data),
        train_context_start=train_context_start,
        train_context_count=train_context_count,
        answer_cache_dir=args.answer_cache_dir,
        prefill_chunk=prefill_chunk,
        retention_scheduler=str(retention_scheduler),
        retention_min=retention_min,
        retention_max=retention_max,
        validation_retention_ratio=validation_ratio,
        ste_temperature=ste_temperature,
        gate_dim=gate_dim,
        gate_sink=gate_sink,
        compute_dtype=None if compute_dtype is None else str(compute_dtype),
        graph_dim=graph_dim,
        gram_normalization=str(gram_normalization),
        leaky_relu_slope=leaky_relu_slope,
        alpha_init=float(alpha_init),
        graph_microbatch_size=graph_microbatch_size,
        token_microbatch_size=token_microbatch_size,
        subgraph_size=subgraph_size,
        gate_lr=gate_lr,
        mixer_lr=mixer_lr,
        weight_decay=weight_decay,
        adamw_eps=adamw_eps,
        amsgrad=amsgrad,
        gate_scheduler=gate_scheduler,
        mixer_scheduler=mixer_scheduler,
        save_strategy=str(save_strategy),
        save_every=save_every,
        save_best=save_best,
        eval_strategy=str(eval_strategy),
        eval_every=eval_every,
        seed=seed,
        wandb_mode=args.wandb_mode,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_name=args.wandb_name,
    )


@dataclass(frozen=True)
class DataSplits:
    train_dataset: object
    train_indices: tuple[int, ...]
    validation_dataset: object
    validation_indices: tuple[int, ...]


def load_data_splits(
    options: AnswerTrainingOptions,
    teacher,
    *,
    dataset_loader=None,
    splits_loader=None,
) -> DataSplits:
    if dataset_loader is None or splits_loader is None:
        from data import available_splits, load_dataset_all

        dataset_loader = dataset_loader or load_dataset_all
        splits_loader = splits_loader or available_splits
    common = {
        "teacher": teacher,
        "answer_cache_dir": options.answer_cache_dir,
    }
    training = dataset_loader(
        options.data,
        teacher.tokenizer,
        split="train",
        start=options.train_context_start,
        count=options.train_context_count,
        **common,
    )
    if "validation" in splits_loader(options.data):
        validation_count = max(1, math.ceil(options.train_context_count * 0.1))
        validation = dataset_loader(
            options.data,
            teacher.tokenizer,
            split="validation",
            start=0,
            count=validation_count,
            **common,
        )
        if not len(training) or not len(validation):
            raise ValueError("native training and validation splits must be non-empty")
        return DataSplits(
            training,
            tuple(range(len(training))),
            validation,
            tuple(range(len(validation))),
        )

    train_indices, validation_indices = fallback_validation_split(range(len(training)))
    return DataSplits(training, train_indices, training, validation_indices)


def initial_cursor(*, total_steps: int, retention_rng: random.Random):
    _positive_int("retention horizon", total_steps)
    return {
        "epoch": 0,
        "phase": "train",
        "offset": 0,
        "optimizer_step": 0,
        "retention_horizon": total_steps,
        "retention_rng_state": retention_rng.getstate(),
        "training_tokens": 0,
        "best_validation_nll": float("inf"),
    }


def scheduled_retention_ratio(options, cursor, rng: random.Random) -> float:
    kind = options.retention_scheduler
    return retention_ratio(
        kind,
        options.retention_min,
        options.retention_max,
        global_step=int(cursor["optimizer_step"]),
        total_steps=int(cursor["retention_horizon"]),
        rng=rng if kind == "uniform" else None,
    )


def advance_cursor(
    cursor,
    *,
    token_count: int,
    contexts_per_epoch: int,
    retention_rng: random.Random,
):
    _positive_int("contexts per epoch", contexts_per_epoch)
    cursor = copy.deepcopy(cursor)
    if cursor.get("phase") != "train":
        raise ValueError("checkpoint cursor must be at a training example")
    cursor["training_tokens"] += int(token_count)
    cursor["offset"] += 1
    cursor["retention_rng_state"] = retention_rng.getstate()
    completed_epoch = cursor["offset"] == contexts_per_epoch
    if completed_epoch:
        cursor["offset"] = 0
        cursor["epoch"] += 1
    return cursor, completed_epoch


def epoch_training_indices(selection, *, seed: int, epoch: int, shuffle: bool):
    indices = list(selection.train_indices)
    if shuffle:
        random.Random(seed + epoch).shuffle(indices)
    return indices


def processed_examples(cursor, contexts_per_epoch):
    return int(cursor["epoch"]) * contexts_per_epoch + int(cursor["offset"])


@dataclass(frozen=True)
class RuntimeState:
    cursor: dict[str, object]
    prefix_ids: torch.Tensor | None
    retention_rng: random.Random


def restore_training_state(
    payload,
    *,
    initialization: str,
    scorer,
    gate_optimizer,
    mixer_optimizer,
    gate_scheduler,
    mixer_scheduler,
    total_steps: int,
    seed: int,
) -> RuntimeState:
    rng = random.Random(seed)
    if initialization == "fresh":
        if payload is not None:
            raise ValueError("fresh initialization cannot receive a checkpoint")
        return RuntimeState(initial_cursor(total_steps=total_steps, retention_rng=rng), None, rng)
    if payload is None:
        raise ValueError("checkpoint initialization requires a payload")
    if initialization == "resume":
        load_checkpoint(
            payload,
            scorer=scorer,
            gate_optimizer=gate_optimizer,
            mixer_optimizer=mixer_optimizer,
            gate_scheduler=gate_scheduler,
            mixer_scheduler=mixer_scheduler,
            restore_rng=True,
        )
        cursor = copy.deepcopy(payload["data_cursor"])
        if "optimizer_step" not in cursor:
            cursor["optimizer_step"] = cursor["global_step"]
        cursor.pop("global_step", None)
        cursor.pop("wandb_step", None)
        if cursor.get("retention_horizon") != total_steps:
            raise ValueError("resume retention horizon conflicts with current data")
        rng.setstate(cursor["retention_rng_state"])
    elif initialization == "graph-checkpoint":
        load_checkpoint(payload, scorer=scorer, restore_rng=False)
        cursor = initial_cursor(total_steps=total_steps, retention_rng=rng)
    else:
        raise ValueError("unknown initialization mode")
    prefix = payload["prefix_ids"].detach().to("cpu").clone()
    return RuntimeState(cursor, prefix, rng)


@dataclass(frozen=True)
class AnswerStepResult:
    answer_nll: float
    answer_nll_sum: float
    answer_token_accuracy: float
    correct_tokens: int
    answer_tokens: int
    context_tokens: int
    grad_norm: float
    gate_grad_norm: float
    mixer_grad_norm: float
    score_grad_norm: float
    retained_score_grad_norm: float
    evicted_score_grad_norm: float
    prefix_ids: torch.Tensor | None

    @classmethod
    def validation(cls, *, nll_sum, correct_tokens, answer_tokens, context_tokens=0):
        return cls(
            answer_nll=float(nll_sum) / answer_tokens,
            answer_nll_sum=float(nll_sum),
            answer_token_accuracy=correct_tokens / answer_tokens,
            correct_tokens=correct_tokens,
            answer_tokens=answer_tokens,
            context_tokens=context_tokens,
            grad_norm=0.0,
            gate_grad_norm=0.0,
            mixer_grad_norm=0.0,
            score_grad_norm=0.0,
            retained_score_grad_norm=0.0,
            evicted_score_grad_norm=0.0,
            prefix_ids=None,
        )


def _normal_prefix(kv) -> torch.Tensor:
    prefix = kv.prefill_ids[:, : int(kv.start_idx)]
    with torch.inference_mode(False):
        return prefix.detach().to("cpu", copy=True)


def _prepare_answer(wrapper, index: int, options, expected_prefix):
    kv = wrapper.prefill_context(
        index,
        prefill_chunk=options.prefill_chunk,
        save_hidden=True,
        do_score=False,
    )
    prefix = _normal_prefix(kv)
    if expected_prefix is not None and not torch.equal(prefix, expected_prefix):
        raise ValueError("context prefix differs from the checkpointed prefix")
    row = wrapper.dataset[index]
    answers = row.get("answers")
    if answers is None:
        answers = wrapper.dataset.resolve_answers(index, kv)
    questions = row.get("question")
    if not questions or not answers:
        raise ValueError("answer training requires a question and teacher answer")
    from data.wrapper import get_query

    query_ids = wrapper.model.apply_template(get_query("qa", questions[0]))
    answer_ids = wrapper.model.encode(answers[0])
    input_ids = torch.cat((query_ids, answer_ids), dim=1)
    start, end = int(kv.start_idx), int(kv.end_idx)
    hidden = tuple(layer[:, start:end] for layer in kv.hidden_cache)
    return kv, hidden, input_ids, query_ids.size(1), prefix


def install_compacted_cache(full_kv, compacted):
    """Create a one-use physical DynamicCache copy with coherent seen length."""

    cache = copy.copy(full_kv)
    cache.key_cache = list(compacted.keys)
    cache.value_cache = list(compacted.values)
    cache._seen_tokens = int(getattr(full_kv, "_seen_tokens", full_kv.end_idx))
    cache.start_idx = int(full_kv.start_idx)
    cache.end_idx = int(full_kv.end_idx)
    cache.ctx_len = int(compacted.indices.size(-1))
    cache.sink = cache.start_idx
    cache.prefill_ids = None
    cache.ctx_ids = None
    cache.hidden_cache = []
    cache.score = None
    cache.valid = None
    cache.flatten = False
    cache.protected_window = 0
    return cache


def _step_scheduler(scheduler) -> None:
    if scheduler is not None and not isinstance(
        scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
    ):
        scheduler.step()


def _gradient_norms(scorer):
    gate, mixer = _model_gradient_norms(scorer)
    return (
        float(torch.sqrt(gate.square() + mixer.square()).mean().item()),
        float(gate.mean().item()),
        float(mixer.mean().item()),
    )


def train_answer_example(
    wrapper,
    index: int,
    *,
    scorer,
    options,
    ratio: float,
    expected_prefix,
) -> AnswerStepResult:
    """Add one question's mean-loss gradient, leaving optimizer steps to the batch."""
    full_kv, hidden, input_ids, answer_start, prefix = _prepare_answer(
        wrapper, index, options, expected_prefix
    )
    # This scorer pass is replayed after LLM backward to avoid retaining its
    # activations; both passes must remain deterministic.
    with torch.no_grad():
        initial_scores = score_context_subgraphs(
            scorer,
            hidden,
            subgraph_size=options.subgraph_size,
            token_microbatch_size=options.token_microbatch_size,
            graph_microbatch_size=options.graph_microbatch_size,
        )
    raw_scores = initial_scores.detach().requires_grad_(True)
    compacted = compact_context_kv(
        full_kv.key_cache,
        full_kv.value_cache,
        raw_scores,
        ratio=ratio,
        temperature=options.ste_temperature,
        context_range=(int(full_kv.start_idx), int(full_kv.end_idx)),
    )
    cache = install_compacted_cache(full_kv, compacted)
    outputs = wrapper.model(
        input_ids,
        cache,
        update_cache=True,
        return_logits=True,
        use_cache=True,
        cache_position=torch.arange(
            cache._seen_tokens,
            cache._seen_tokens + input_ids.size(1),
            device=input_ids.device,
        ),
    )
    objective = answer_objective(outputs.logits, input_ids, answer_start=answer_start)
    objective.loss.backward()
    health = replay_score_gradients(
        scorer,
        hidden,
        raw_scores.grad,
        compacted.indices,
        subgraph_size=options.subgraph_size,
        token_microbatch_size=options.token_microbatch_size,
        graph_microbatch_size=options.graph_microbatch_size,
    )
    return AnswerStepResult(
        answer_nll=float(objective.loss.detach().item()),
        answer_nll_sum=float(objective.nll_sum.detach().item()),
        answer_token_accuracy=objective.accuracy,
        correct_tokens=objective.correct_tokens,
        answer_tokens=objective.token_count,
        context_tokens=int(full_kv.ctx_len),
        grad_norm=0.0,
        gate_grad_norm=0.0,
        mixer_grad_norm=0.0,
        score_grad_norm=float(health.overall.item()),
        retained_score_grad_norm=float(health.retained.item()),
        evicted_score_grad_norm=float(health.evicted.item()),
        prefix_ids=prefix,
    )


def finish_answer_batch(
    results: Sequence[AnswerStepResult], *, scorer, gate_optimizer, mixer_optimizer,
    gate_scheduler, mixer_scheduler,
) -> AnswerStepResult:
    """Average the accumulated question gradients and perform one update."""
    count = _positive_int("batch examples", len(results))
    for parameter in scorer.parameters():
        if parameter.grad is not None:
            parameter.grad.div_(count)
    grad_norm, gate_grad_norm, mixer_grad_norm = _gradient_norms(scorer)
    gate_optimizer.step()
    mixer_optimizer.step()
    _step_scheduler(gate_scheduler)
    _step_scheduler(mixer_scheduler)
    means = {
        key: sum(getattr(result, key) for result in results) / count
        for key in (
            "answer_nll", "answer_token_accuracy", "score_grad_norm",
            "retained_score_grad_norm", "evicted_score_grad_norm",
        )
    }
    totals = {
        key: sum(getattr(result, key) for result in results)
        for key in ("answer_nll_sum", "correct_tokens", "answer_tokens", "context_tokens")
    }
    return replace(
        results[-1], **means, **totals, grad_norm=grad_norm,
        gate_grad_norm=gate_grad_norm, mixer_grad_norm=mixer_grad_norm,
    )


def evaluate_answer_example(
    wrapper,
    index: int,
    *,
    scorer,
    options,
    expected_prefix,
) -> AnswerStepResult:
    full_kv, hidden, input_ids, answer_start, _ = _prepare_answer(
        wrapper, index, options, expected_prefix
    )
    with torch.inference_mode():
        scores = score_context_subgraphs(
            scorer,
            hidden,
            subgraph_size=options.subgraph_size,
            token_microbatch_size=options.token_microbatch_size,
            graph_microbatch_size=options.graph_microbatch_size,
        )
        compacted = compact_context_kv(
            full_kv.key_cache,
            full_kv.value_cache,
            scores,
            ratio=options.validation_retention_ratio,
            temperature=options.ste_temperature,
            context_range=(int(full_kv.start_idx), int(full_kv.end_idx)),
            straight_through=False,
        )
        cache = install_compacted_cache(full_kv, compacted)
        outputs = wrapper.model(
            input_ids,
            cache,
            update_cache=True,
            return_logits=True,
            use_cache=True,
            cache_position=torch.arange(
                cache._seen_tokens,
                cache._seen_tokens + input_ids.size(1),
                device=input_ids.device,
            ),
        )
        objective = answer_objective(outputs.logits, input_ids, answer_start=answer_start)
    return AnswerStepResult.validation(
        nll_sum=float(objective.nll_sum.item()),
        correct_tokens=objective.correct_tokens,
        answer_tokens=objective.token_count,
        context_tokens=int(full_kv.ctx_len),
    )


@dataclass(frozen=True)
class ValidationAggregate:
    answer_nll: float
    answer_token_accuracy: float


def aggregate_validation(results: Sequence[AnswerStepResult]) -> ValidationAggregate:
    token_count = sum(result.answer_tokens for result in results)
    if token_count < 1:
        raise ValueError("validation requires at least one answer token")
    return ValidationAggregate(
        sum(result.answer_nll_sum for result in results) / token_count,
        sum(result.correct_tokens for result in results) / token_count,
    )


def update_validation_cursor(cursor, metrics) -> tuple[dict[str, object], bool]:
    cursor = copy.deepcopy(cursor)
    previous = float(cursor["best_validation_nll"])
    improved = float(metrics.answer_nll) < previous
    if improved:
        cursor["best_validation_nll"] = float(metrics.answer_nll)
    return cursor, improved


def train_log_metrics(
    result,
    *,
    scorer,
    gate_optimizer,
    mixer_optimizer,
    fractional_epoch,
    cumulative_tokens,
    examples,
    optimizer_step,
    batch_examples,
):
    metrics = {
        "train/answer_nll": result.answer_nll,
        "train/answer_token_accuracy": result.answer_token_accuracy,
        "train/grad_norm": result.grad_norm,
        "train/gate_grad_norm": result.gate_grad_norm,
        "train/mixer_grad_norm": result.mixer_grad_norm,
        "train/score_grad_norm": result.score_grad_norm,
        "train/retained_score_grad_norm": result.retained_score_grad_norm,
        "train/evicted_score_grad_norm": result.evicted_score_grad_norm,
        "train/mean_alpha": float(scorer.mixer.alpha.detach().float().mean().item()),
        "train/gate_learning_rate": train_graph._optimizer_lr(gate_optimizer),
        "train/mixer_learning_rate": train_graph._optimizer_lr(mixer_optimizer),
        "train/epoch": float(fractional_epoch),
        "train/tokens": int(cumulative_tokens),
        "train/examples": int(examples),
        "train/optimizer_step": int(optimizer_step),
        "train/batch_examples": int(batch_examples),
    }
    if set(metrics) != TRAIN_LOG_KEYS:
        raise AssertionError("answer-training W&B train metric allowlist changed")
    return metrics


def validation_log_metrics(result):
    metrics = {
        "validation/answer_nll": float(result.answer_nll),
        "validation/answer_token_accuracy": float(result.answer_token_accuracy),
    }
    if set(metrics) != VALIDATION_LOG_KEYS:
        raise AssertionError("answer-training W&B validation metric allowlist changed")
    return metrics


def answer_checkpoint_config(base_config, *, options, total_steps: int):
    config = copy.deepcopy(dict(base_config))
    config.update(
        {
            "objective": OBJECTIVE,
            "dataset": options.data,
            "train_context_start": options.train_context_start,
            "train_context_count": options.train_context_count,
            "deduplication_version": DEDUPLICATION_VERSION,
            "answer_generation_protocol": ANSWER_GENERATION_PROTOCOL,
            "answer_cache_identity_protocol": ANSWER_CACHE_IDENTITY_PROTOCOL,
            "ste_temperature": options.ste_temperature,
            "retention_scheduler": options.retention_scheduler,
            "retention_min": options.retention_min,
            "retention_max": options.retention_max,
            "retention_horizon": total_steps,
            "validation_retention_ratio": options.validation_retention_ratio,
            "epochs": options.epochs,
            "gradient_accumulation_steps": options.gradient_accumulation_steps,
            "shuffle_data": options.shuffle_data,
            "seed": options.seed,
            "weight_decay": options.weight_decay,
            "save_strategy": options.save_strategy,
            "save_every": options.save_every,
            "save_best": options.save_best,
            "eval_strategy": options.eval_strategy,
            "eval_every": options.eval_every,
        }
    )
    return config


def _make_components(teacher, options, *, total_steps):
    model_config, layers, heads, query_groups = train_graph._model_dimensions(teacher)
    microbatch = train_graph.resolve_graph_microbatch_size(
        options.graph_microbatch_size, layers, heads
    )
    options = replace(options, graph_microbatch_size=microbatch)
    gates = train_graph._random_gates(teacher, model_config, options)
    scorer = ImplicitGraphScorer(
        gates,
        teacher.config,
        graph_dim=options.graph_dim,
        graph_microbatch_size=microbatch,
        gram_normalization=options.gram_normalization,
        leaky_relu_slope=options.leaky_relu_slope,
        alpha_init=options.alpha_init,
        compute_dtype=(
            None
            if options.compute_dtype is None
            else parse_compute_dtype(options.compute_dtype)
        ),
    )
    gate_optimizer, mixer_optimizer = build_adamw_optimizers(
        scorer,
        gate_lr=options.gate_lr,
        mixer_lr=options.mixer_lr,
        weight_decay=options.weight_decay,
        eps=options.adamw_eps,
        amsgrad=options.amsgrad,
    )
    gate_scheduler = build_scheduler(
        gate_optimizer, options.gate_scheduler, total_steps=total_steps
    )
    mixer_scheduler = build_scheduler(
        mixer_optimizer, options.mixer_scheduler, total_steps=total_steps
    )
    base = train_graph.normalized_checkpoint_config(
        model_id=options.model_id,
        scorer=scorer,
        options=options,
        query_groups=query_groups,
    )
    config = answer_checkpoint_config(base, options=options, total_steps=total_steps)
    return (
        options,
        scorer,
        gate_optimizer,
        mixer_optimizer,
        gate_scheduler,
        mixer_scheduler,
        config,
    )


def _step_plateau(schedulers, validation_nll: float) -> None:
    for scheduler in schedulers:
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(validation_nll)


def answer_cadence_due(strategy, every, *, cursor, completed_epoch):
    if strategy == "steps":
        return int(cursor["optimizer_step"]) % every == 0
    return completed_epoch and int(cursor["epoch"]) % every == 0


def run_training(
    args,
    *,
    model_factory=None,
    dataset_loader=None,
    splits_loader=None,
    wrapper_factory=None,
    wandb_module=wandb,
    progress_factory=tqdm,
):
    checkpoint = None
    checkpoint_path = args.resume or args.graph_checkpoint
    if checkpoint_path is not None:
        from graph.evaluation import load_evaluation_checkpoint

        checkpoint = load_evaluation_checkpoint(
            checkpoint_path, model_override=getattr(args, "model", None)
        )
    payload = None if checkpoint is None else checkpoint.payload
    options = resolve_options(args, payload)
    checkpoint_run_id = (
        payload.get("wandb_run_id") if options.initialization == "resume" else None
    )
    run_id = train_graph._persistent_wandb_run_id(
        options, wandb_module, checkpoint_run_id=checkpoint_run_id
    )
    run = train_graph._initialize_wandb(options, wandb_module, run_id=run_id)
    succeeded = False
    progress = None
    try:
        train_graph._set_seed(options.seed)
        if model_factory is None:
            from model import ModelKVzip

            model_factory = ModelKVzip
        teacher = model_factory(
            options.model_id, kv_type="retain", gate_path_or_name=""
        )
        freeze_llm(teacher.model)
        selection = load_data_splits(
            options,
            teacher,
            dataset_loader=dataset_loader,
            splits_loader=splits_loader,
        )
        contexts_per_epoch = len(selection.train_indices)
        updates_per_epoch = math.ceil(contexts_per_epoch / options.gradient_accumulation_steps)
        total_steps = options.epochs * updates_per_epoch
        (
            options,
            scorer,
            gate_optimizer,
            mixer_optimizer,
            gate_scheduler,
            mixer_scheduler,
            checkpoint_config,
        ) = _make_components(teacher, options, total_steps=total_steps)

        if wrapper_factory is None:
            from data import DataWrapper

            wrapper_factory = DataWrapper
        train_wrapper = wrapper_factory(options.data, selection.train_dataset, teacher)
        validation_wrapper = (
            train_wrapper
            if selection.validation_dataset is selection.train_dataset
            else wrapper_factory(options.data, selection.validation_dataset, teacher)
        )

        if options.initialization == "resume":
            train_graph._validate_resume_config(
                normalized_answer_resume_config(payload["config"]), checkpoint_config
            )
        state = restore_training_state(
            payload,
            initialization=options.initialization,
            scorer=scorer,
            gate_optimizer=gate_optimizer,
            mixer_optimizer=mixer_optimizer,
            gate_scheduler=gate_scheduler,
            mixer_scheduler=mixer_scheduler,
            total_steps=total_steps,
            seed=options.seed,
        )
        cursor = state.cursor
        retention_rng = state.retention_rng
        training_prefix = state.prefix_ids
        if training_prefix is not None:
            from graph.evaluation import restore_checkpoint_prefix

            restore_checkpoint_prefix(teacher, training_prefix)

        if hasattr(run, "config"):
            run.config.update(
                {**checkpoint_config, "prefill_chunk": options.prefill_chunk},
                allow_val_change=True,
            )
        # Resumed runs may have concrete definitions expanded from the old glob.
        for metric in ["*", *sorted(TRAIN_LOG_KEYS | VALIDATION_LOG_KEYS)]:
            run.define_metric(metric, overwrite=True)
        initial_examples = processed_examples(cursor, contexts_per_epoch)
        progress_total = options.epochs * contexts_per_epoch
        progress = progress_factory(
            total=progress_total,
            initial=initial_examples,
            desc="Answer training",
            unit="example",
            position=1,
        )

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
                gate_optimizer=gate_optimizer,
                mixer_optimizer=mixer_optimizer,
                gate_scheduler=gate_scheduler,
                mixer_scheduler=mixer_scheduler,
            )

        def evaluate():
            nonlocal cursor
            scorer.eval()
            results = [
                evaluate_answer_example(
                    validation_wrapper,
                    index,
                    scorer=scorer,
                    options=options,
                    expected_prefix=training_prefix,
                )
                for index in selection.validation_indices
            ]
            scorer.train()
            aggregate = aggregate_validation(results)
            _step_plateau(
                (gate_scheduler, mixer_scheduler), aggregate.answer_nll
            )
            cursor, improved = update_validation_cursor(cursor, aggregate)
            return validation_log_metrics(aggregate), improved

        processed = 0
        last_saved = True
        order_epoch = None
        scorer.train()
        while processed_examples(cursor, contexts_per_epoch) < progress_total:
            if order_epoch != int(cursor["epoch"]):
                order_epoch = int(cursor["epoch"])
                order = epoch_training_indices(
                    selection, seed=options.seed, epoch=order_epoch, shuffle=options.shuffle_data
                )
            offset = int(cursor["offset"])
            batch_indices = order[offset : offset + options.gradient_accumulation_steps]
            gate_optimizer.zero_grad(set_to_none=True)
            mixer_optimizer.zero_grad(set_to_none=True)
            results = []
            for index in batch_indices:
                ratio = scheduled_retention_ratio(options, cursor, retention_rng)
                result = train_answer_example(
                    train_wrapper, index, scorer=scorer, options=options, ratio=ratio,
                    expected_prefix=training_prefix,
                )
                if training_prefix is None:
                    training_prefix = result.prefix_ids
                results.append(result)
                cursor, completed_epoch = advance_cursor(
                    cursor, token_count=result.context_tokens,
                    contexts_per_epoch=contexts_per_epoch, retention_rng=retention_rng,
                )
                processed += 1
                progress.update(1)
            result = finish_answer_batch(
                results, scorer=scorer, gate_optimizer=gate_optimizer,
                mixer_optimizer=mixer_optimizer, gate_scheduler=gate_scheduler,
                mixer_scheduler=mixer_scheduler,
            )
            cursor["optimizer_step"] += 1
            metrics = train_log_metrics(
                result,
                scorer=scorer,
                gate_optimizer=gate_optimizer,
                mixer_optimizer=mixer_optimizer,
                fractional_epoch=processed_examples(cursor, contexts_per_epoch) / contexts_per_epoch,
                cumulative_tokens=int(cursor["training_tokens"]),
                examples=processed_examples(cursor, contexts_per_epoch),
                optimizer_step=int(cursor["optimizer_step"]),
                batch_examples=len(results),
            )
            last_saved = False
            progress.set_postfix(
                {
                    key.removeprefix("train/"): value
                    for key, value in metrics.items()
                    if key in {"train/answer_nll", "train/answer_token_accuracy"}
                }
            )
            eval_due = answer_cadence_due(
                options.eval_strategy,
                options.eval_every,
                cursor=cursor,
                completed_epoch=completed_epoch,
            )
            save_due = answer_cadence_due(
                options.save_strategy,
                options.save_every,
                cursor=cursor,
                completed_epoch=completed_epoch,
            )
            improved = False
            if eval_due:
                progress.set_description("Validating answers")
                validation_metrics, improved = evaluate()
                metrics.update(validation_metrics)
                progress.set_description("Answer training")
            run.log(metrics, step=int(cursor["optimizer_step"]), commit=True)
            if options.save_best and improved:
                save("best")
            if save_due:
                save("last")
                last_saved = True
        if processed and not last_saved:
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
