"""Training primitives for the streamed implicit whole-context mixer."""

from __future__ import annotations

import copy
import json
import math
import os
import random
import tempfile
import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from .model import (
    DEFAULT_INJECTION_INIT,
    DEFAULT_MIXER_ARCHITECTURE,
    DEFAULT_MIXER_COUPLING,
    mixer_activation_order,
    _select_graph_rows,
    GRANOLA_ADAPTIVITY,
    LEGACY_ACTIVATION_ORDER,
    NORMALIZATION_CONFIG_KEYS,
    canonical_normalization_config,
    NORMALIZATION_SHARING,
    NORMALIZATIONS,
    ContextNormStats,
    GraphBatch,
    ImplicitGraphScorer,
    PreparedImplicitGraph,
    _GranolaNormState,
    compute_dtype_name,
    derive_evaluation_rnf_seed,
    parse_compute_dtype,
    resolve_graph_microbatch_size,
    subgraph_groups,
)


def _normal_cpu_tensor(tensor: Tensor) -> Tensor:
    return tensor.detach().to("cpu").clone()


def _normal_hidden_tensor(tensor: Tensor) -> Tensor:
    tensor = _normal_cpu_tensor(tensor)
    if tensor.ndim == 3 and tensor.size(0) == 1:
        tensor = tensor[0]
    return tensor


def _is_normal_cpu_tensor(tensor: Tensor) -> bool:
    return (
        isinstance(tensor, Tensor)
        and tensor.device.type == "cpu"
        and not tensor.requires_grad
        and not tensor.is_inference()
    )


def _validate_example_fields(
    *,
    dataset_name: str,
    dataset_index: int,
    token_ids: Tensor,
    hidden_by_layer: Sequence[Tensor],
    teacher_scores: Tensor,
    prefix_ids: Tensor,
    sequence_length: int,
) -> tuple[Tensor, tuple[Tensor, ...], Tensor, Tensor]:
    if not isinstance(dataset_name, str) or not dataset_name:
        raise ValueError("dataset name must be a non-empty string")
    if isinstance(dataset_index, bool) or not isinstance(dataset_index, int) or dataset_index < 0:
        raise ValueError("dataset index must be a non-negative integer")
    if sequence_length < 1:
        raise ValueError("sequence length must be positive")
    hidden = tuple(hidden_by_layer)
    if not hidden or any(tensor.ndim != 2 for tensor in hidden):
        raise ValueError("hidden tensors must have shape [tokens, hidden_dim]")
    if any(tensor.size(0) != sequence_length for tensor in hidden):
        raise ValueError("hidden tensor sequence length does not match example")
    if token_ids.ndim not in {1, 2} or token_ids.size(-1) != sequence_length:
        raise ValueError("token ID sequence length does not match example")
    if teacher_scores.ndim != 4 or teacher_scores.size(-1) != sequence_length:
        raise ValueError("teacher score sequence length does not match example")
    if teacher_scores.size(0) != len(hidden):
        raise ValueError("teacher score layers do not match hidden tensors")
    if prefix_ids.ndim not in {1, 2}:
        raise ValueError("prefix IDs must be one- or two-dimensional")
    return token_ids, hidden, teacher_scores, prefix_ids


@dataclass(frozen=True)
class TeacherExample:
    """CPU teacher data for exactly one whole context."""

    dataset_name: str
    dataset_index: int
    token_ids: Tensor
    hidden_by_layer: Sequence[Tensor]
    teacher_scores: Tensor
    prefix_ids: Tensor
    sequence_length: int

    def __post_init__(self) -> None:
        hidden = tuple(_normal_hidden_tensor(tensor) for tensor in self.hidden_by_layer)
        token_ids = _normal_cpu_tensor(self.token_ids)
        teacher_scores = _normal_cpu_tensor(self.teacher_scores)
        prefix_ids = _normal_cpu_tensor(self.prefix_ids)
        token_ids, hidden, teacher_scores, prefix_ids = _validate_example_fields(
            dataset_name=self.dataset_name,
            dataset_index=self.dataset_index,
            token_ids=token_ids,
            hidden_by_layer=hidden,
            teacher_scores=teacher_scores,
            prefix_ids=prefix_ids,
            sequence_length=self.sequence_length,
        )
        object.__setattr__(self, "hidden_by_layer", hidden)
        object.__setattr__(self, "token_ids", token_ids)
        object.__setattr__(self, "teacher_scores", teacher_scores)
        object.__setattr__(self, "prefix_ids", prefix_ids)

    @classmethod
    def from_owned_cpu(
        cls,
        *,
        dataset_name: str,
        dataset_index: int,
        token_ids: Tensor,
        hidden_by_layer: Sequence[Tensor],
        teacher_scores: Tensor,
        prefix_ids: Tensor,
        sequence_length: int,
    ) -> TeacherExample:
        """Build without copying already-normal CPU tensors owned by the caller."""

        values = (token_ids, *hidden_by_layer, teacher_scores, prefix_ids)
        if not all(_is_normal_cpu_tensor(value) for value in values):
            raise ValueError("owned teacher tensors must be normal CPU tensors without gradients")
        hidden = tuple(
            value[0] if value.ndim == 3 and value.size(0) == 1 else value
            for value in hidden_by_layer
        )
        token_ids, hidden, teacher_scores, prefix_ids = _validate_example_fields(
            dataset_name=dataset_name,
            dataset_index=dataset_index,
            token_ids=token_ids,
            hidden_by_layer=hidden,
            teacher_scores=teacher_scores,
            prefix_ids=prefix_ids,
            sequence_length=sequence_length,
        )
        instance = object.__new__(cls)
        object.__setattr__(instance, "dataset_name", dataset_name)
        object.__setattr__(instance, "dataset_index", dataset_index)
        object.__setattr__(instance, "token_ids", token_ids)
        object.__setattr__(instance, "hidden_by_layer", hidden)
        object.__setattr__(instance, "teacher_scores", teacher_scores)
        object.__setattr__(instance, "prefix_ids", prefix_ids)
        object.__setattr__(instance, "sequence_length", sequence_length)
        return instance


@dataclass(frozen=True)
class SchedulerSpec:
    name: str
    kwargs: dict[str, object]


def _scheduler_class(name: str):
    scheduler_class = getattr(torch.optim.lr_scheduler, name, None)
    if not isinstance(scheduler_class, type) or not issubclass(
        scheduler_class,
        (torch.optim.lr_scheduler.LRScheduler, torch.optim.lr_scheduler.ReduceLROnPlateau),
    ):
        raise ValueError(f"unknown PyTorch scheduler: {name}")
    return scheduler_class


def parse_scheduler_spec(
    name: str | None, kwargs: str | Mapping[str, object] | None = None
) -> SchedulerSpec | None:
    """Parse and instantiate-check a scheduler before loading the model."""

    if name is None:
        if kwargs is not None:
            raise ValueError("scheduler kwargs were supplied without a scheduler")
        return None
    try:
        parsed = json.loads(kwargs) if isinstance(kwargs, str) else dict(kwargs or {})
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("scheduler kwargs must be a JSON object") from error
    if not isinstance(parsed, dict):
        raise ValueError("scheduler kwargs must be a JSON object")
    try:
        parsed = json.loads(json.dumps(parsed, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError("scheduler kwargs must be a JSON object") from error
    if name == "LinearWarmupCosineLR":
        fraction = parsed.get("warmup_fraction")
        if (
            set(parsed) != {"warmup_fraction"}
            or isinstance(fraction, bool)
            or not isinstance(fraction, Real)
            or not math.isfinite(fraction)
            or not 0 < fraction < 1
        ):
            raise ValueError("warmup_fraction must be between 0 and 1")
        return SchedulerSpec(name, parsed)
    scheduler_class = _scheduler_class(name)
    parameter = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.AdamW([parameter], lr=1e-3)
    try:
        scheduler_class(optimizer, **parsed)
    except Exception as error:
        raise ValueError(f"invalid {name} scheduler arguments: {error}") from error
    return SchedulerSpec(name, parsed)


def build_scheduler(optimizer, spec: SchedulerSpec | None, *, total_steps=None):
    if spec is None:
        return None
    if spec.name == "LinearWarmupCosineLR":
        if (
            isinstance(total_steps, bool)
            or not isinstance(total_steps, int)
            or total_steps < 2
        ):
            raise ValueError("LinearWarmupCosineLR requires at least two total steps")
        warmup_steps = min(
            total_steps - 1,
            max(1, round(total_steps * spec.kwargs["warmup_fraction"])),
        )

        def lr_factor(step):
            if step < warmup_steps:
                return (step + 1) / warmup_steps
            progress = (step - warmup_steps + 1) / (total_steps - warmup_steps + 1)
            return (1 + math.cos(math.pi * progress)) / 2

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    return _scheduler_class(spec.name)(optimizer, **spec.kwargs)


def _valid_lr(value: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError("learning rates must be finite and positive")
    return float(value)


class PhaseTiming:
    """Collect phase timings without synchronizing inside staged training."""

    def __init__(self, device=None, *, clock=time.perf_counter) -> None:
        self.device = torch.device(device or "cpu")
        self.clock = clock
        self._cuda = self.device.type == "cuda" and torch.cuda.is_available()
        self._cpu_seconds = defaultdict(float)
        self._cuda_events = []

    @contextmanager
    def region(self, phase: str, operation: str):
        if phase not in {"gate", "graph", "joint"} or operation not in {"forward", "backward"}:
            raise ValueError("timing region must be gate/graph/joint and forward/backward")
        key = f"{phase}_{operation}_seconds"
        if self._cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            try:
                yield
            finally:
                end.record()
                self._cuda_events.append((key, start, end))
        else:
            start = self.clock()
            try:
                yield
            finally:
                self._cpu_seconds[key] += self.clock() - start

    def resolve(self) -> dict[str, float]:
        result = dict(self._cpu_seconds)
        if self._cuda_events:
            torch.cuda.synchronize(self.device)
            for key, start, end in self._cuda_events:
                result[key] = result.get(key, 0.0) + start.elapsed_time(end) / 1000
        self._cpu_seconds.clear()
        self._cuda_events.clear()
        return result


def build_adamw_optimizers(
    scorer: ImplicitGraphScorer,
    *,
    gate_lr: float = 1e-4,
    mixer_lr: float = 1e-3,
    weight_decay: float = 0.01,
    eps: float = 1e-8,
    amsgrad: bool = False,
    gate_frozen: bool = False,
    mixer_frozen: bool | None = None,
):
    """Return disjoint gate and mixer AdamW optimizers.

    `mixer_frozen` defaults to None meaning "caller said nothing", which a
    gate-only scorer satisfies by having no mixer. An explicit False asks for a
    trainable mixer, so a scorer without one is a caller error and is raised
    here rather than surfacing an epoch later as a missing optimizer.
    """

    gate_lr, mixer_lr = _valid_lr(gate_lr), _valid_lr(mixer_lr)
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("weight decay must be finite and non-negative")
    gate_parameters = list(scorer.gates.parameters())
    mixer = scorer.mixer
    if mixer is None:
        if mixer_frozen is False:
            raise ValueError(
                "mixer_frozen=False asks for a trainable mixer, but this scorer "
                "has none; build it with a graph_dim to train one"
            )
        mixer_frozen = True
    mixer_frozen = bool(mixer_frozen)
    # Each architecture decides which of its parameters weight decay applies to;
    # normalization scales, shifts, and the residual weight stay undecayed.
    decay_parameters, no_decay_parameters = (
        ([], []) if mixer is None else mixer.parameter_groups()
    )
    for parameter in gate_parameters:
        parameter.requires_grad_(not gate_frozen)
    for parameter in (*decay_parameters, *no_decay_parameters):
        parameter.requires_grad_(not mixer_frozen)
    gate_optimizer = None
    mixer_optimizer = None
    if not gate_frozen:
        gate_optimizer = torch.optim.AdamW(
            gate_parameters,
            lr=gate_lr,
            weight_decay=weight_decay,
            eps=eps,
            amsgrad=amsgrad,
        )
    if not mixer_frozen:
        mixer_optimizer = torch.optim.AdamW(
            [
                {"params": decay_parameters, "weight_decay": weight_decay},
                {"params": no_decay_parameters, "weight_decay": 0.0},
            ],
            lr=mixer_lr,
            eps=eps,
            amsgrad=amsgrad,
        )
    return gate_optimizer, mixer_optimizer


def _cpu_state(state: Mapping[str, Tensor]) -> dict[str, Tensor]:
    return {name: value.detach().to("cpu").clone() for name, value in state.items()}


def _capture_rng_state() -> dict[str, object]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: Mapping[str, object]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    output_dir,
    kind: str,
    *,
    scorer: ImplicitGraphScorer,
    config,
    model_id: str,
    prefix_ids: Tensor,
    prefill_chunk: int,
    data_cursor,
    wandb_run_id: str | None,
    gate_optimizer=None,
    mixer_optimizer=None,
    gate_scheduler=None,
    mixer_scheduler=None,
) -> Path:
    if kind not in {"best", "last"}:
        raise ValueError("checkpoint kind must be best or last")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_config = copy.deepcopy(dict(config))
    dtype_name = compute_dtype_name(scorer.compute_dtype)
    if "compute_dtype" in checkpoint_config and checkpoint_config["compute_dtype"] != dtype_name:
        raise ValueError("checkpoint compute dtype conflicts with scorer")
    checkpoint_config["compute_dtype"] = dtype_name
    full_state = scorer.state_dict()
    payload = {
        "mixer": _cpu_state(
            {name: value for name, value in full_state.items() if not name.startswith("gates.")}
        ),
        "gate": _cpu_state(scorer.gates.state_dict()),
        "mixer_optimizer": None if mixer_optimizer is None else mixer_optimizer.state_dict(),
        "gate_optimizer": None if gate_optimizer is None else gate_optimizer.state_dict(),
        "mixer_scheduler": None if mixer_scheduler is None else mixer_scheduler.state_dict(),
        "gate_scheduler": None if gate_scheduler is None else gate_scheduler.state_dict(),
        "config": checkpoint_config,
        "model_id": model_id,
        "prefix_ids": _normal_cpu_tensor(prefix_ids),
        "prefill_chunk": prefill_chunk,
        "data_cursor": copy.deepcopy(data_cursor),
        "rng": _capture_rng_state(),
        "wandb_run_id": wandb_run_id,
    }
    path = output_dir / f"{kind}.pt"
    with tempfile.NamedTemporaryFile(
        dir=output_dir, prefix=f".{kind}.", suffix=".tmp", delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return path


def _restore_optional_state(component, state, name: str) -> None:
    if component is not None:
        if state is None:
            raise ValueError(f"checkpoint has no {name} state")
        component.load_state_dict(state)


def _checkpoint_normalization_config(
    config: Mapping[str, object], *, graph_dim: int, architecture: str
) -> dict[str, object]:
    legacy = "normalization" not in config
    # The marker to expect comes from the architecture the scorer runs, not the
    # one the checkpoint names: each architecture applies its activation in its
    # own place, and this is the check that a file built for the other one is
    # refused here rather than deep inside loading its weights by name.
    expected = mixer_activation_order(architecture)
    if legacy:
        marker = config.get("activation_order")
        # Checkpoints predating the normalization setting are all implicit
        # mixers, so their older marker is tolerated only for an implicit
        # scorer. A gps scorer has no such checkpoints to accept.
        tolerated = {expected}
        if architecture == "implicit":
            tolerated.add(LEGACY_ACTIVATION_ORDER)
        if marker not in tolerated:
            raise ValueError("checkpoint activation order conflicts with scorer")
        # The scorer's own graph width stands in for the checkpoint's, so a
        # legacy config missing it still canonicalizes to a usable RNF width.
        config = {**config, "graph_dim": graph_dim}
    elif config.get("activation_order") != expected:
        raise ValueError("checkpoint activation order conflicts with scorer")
    canonical = canonical_normalization_config(config)
    result = {name: canonical[name] for name in NORMALIZATION_CONFIG_KEYS}
    if result["normalization"] not in NORMALIZATIONS:
        raise ValueError("checkpoint normalization is invalid")
    if result["normalization_sharing"] not in NORMALIZATION_SHARING:
        raise ValueError("checkpoint normalization sharing is invalid")
    if result["granola_adaptivity"] not in GRANOLA_ADAPTIVITY:
        raise ValueError("checkpoint granola adaptivity is invalid")
    for name in ("granola_gnn_depth", "granola_mlp_depth", "granola_rnf_dim"):
        value = result[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"checkpoint {name} must be a positive integer")
    seed = result["normalization_seed"]
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or not 0 <= seed < 2**63
    ):
        raise ValueError("checkpoint normalization_seed is invalid")
    return result


def load_checkpoint(
    path_or_payload,
    *,
    scorer: ImplicitGraphScorer,
    gate_optimizer=None,
    mixer_optimizer=None,
    gate_scheduler=None,
    mixer_scheduler=None,
    restore_rng: bool = True,
):
    payload = (
        path_or_payload
        if isinstance(path_or_payload, Mapping)
        else torch.load(path_or_payload, map_location="cpu", weights_only=False)
    )
    config = payload.get("config")
    if not isinstance(config, Mapping) or parse_compute_dtype(config.get("compute_dtype")) != scorer.compute_dtype:
        raise ValueError("checkpoint compute dtype conflicts with scorer")
    # A gate-only scorer has no mixer, so it has no normalization to agree on.
    if scorer.mixer is not None:
        saved_normalization = _checkpoint_normalization_config(
            config,
            graph_dim=scorer.graph_dim,
            architecture=scorer.mixer_architecture,
        )
        # Only the settings this mixer actually applies: a GPS stack owns none
        # of the GraNoLa shape, so comparing them would mean nothing.
        expected_normalization = scorer.mixer.normalization_config()
        differing = [
            name
            for name in expected_normalization
            if saved_normalization[name] != expected_normalization[name]
        ]
        if differing:
            raise ValueError(
                "checkpoint normalization configuration conflicts with scorer: "
                + ", ".join(differing)
            )
        # The coupling decides which parameters exist at all, so a mismatch is
        # named here rather than surfacing as a missing key deep in the load.
        saved_coupling = {
            "mixer_coupling": config.get("mixer_coupling", DEFAULT_MIXER_COUPLING)
        }
        for name in ("injection_target", "self_loop_init"):
            if name in config:
                saved_coupling[name] = config[name]
        if "injection_target" in saved_coupling:
            # Checkpoints written before the init scale became a setting
            # started their maps at zero.
            saved_coupling["injection_init"] = config.get(
                "injection_init", DEFAULT_INJECTION_INIT
            )
        expected_coupling = scorer.mixer.coupling_config()
        differing = sorted(
            name
            for name in set(saved_coupling) | set(expected_coupling)
            if saved_coupling.get(name) != expected_coupling.get(name)
        )
        if differing:
            raise ValueError(
                "checkpoint coupling configuration conflicts with scorer: "
                + ", ".join(differing)
            )
    mixer_state, gate_state = payload.get("mixer"), payload.get("gate")
    if not isinstance(mixer_state, Mapping) or not isinstance(gate_state, Mapping):
        raise ValueError("checkpoint must contain mixer and gate state mappings")
    state = dict(mixer_state)
    state.update({f"gates.{name}": value for name, value in gate_state.items()})
    scorer.load_state_dict(state, strict=True)
    _restore_optional_state(mixer_optimizer, payload.get("mixer_optimizer"), "mixer optimizer")
    _restore_optional_state(gate_optimizer, payload.get("gate_optimizer"), "gate optimizer")
    _restore_optional_state(mixer_scheduler, payload.get("mixer_scheduler"), "mixer scheduler")
    _restore_optional_state(gate_scheduler, payload.get("gate_scheduler"), "gate scheduler")
    if restore_rng:
        _restore_rng_state(payload["rng"])
    return payload


def load_gate_checkpoint(scorer: ImplicitGraphScorer, path) -> None:
    payload = torch.load(
        os.path.expanduser(str(path)), map_location="cpu", weights_only=False
    )
    if isinstance(payload, Mapping) and "gate" in payload:
        scorer.gates.load_state_dict(payload["gate"], strict=True)
        return
    states = payload.get("module") if isinstance(payload, Mapping) else None
    if not isinstance(states, Sequence) or len(states) != len(scorer.gates):
        raise ValueError("gate checkpoint must contain one module state per layer")
    for gate, state in zip(scorer.gates, states):
        gate.load_state_dict(state, strict=True)


@contextmanager
def _frozen(parameters):
    parameters = list(parameters)
    states = [parameter.requires_grad for parameter in parameters]
    for parameter in parameters:
        parameter.requires_grad_(False)
    try:
        yield
    finally:
        for parameter, state in zip(parameters, states):
            parameter.requires_grad_(state)


def _gradient_energy(parameter: Tensor, models: int) -> Tensor:
    if parameter.grad is None:
        return torch.zeros(models, device=parameter.device)
    return parameter.grad.detach().float().reshape(models, -1).square().sum(dim=1)


def _mixer_gradient_energy(parameter: Tensor, num_graphs: int) -> Tensor:
    """Attribute one mixer parameter's gradient energy to each layer/head graph.

    Normalization parameters can be shared across graphs, so their leading
    dimension is a group count rather than the graph count. A shared group's
    energy is split evenly across the graphs that use it, the same way the gate
    splits its shared RMSNorm weights across heads.
    """

    if parameter.grad is None:
        return torch.zeros(num_graphs, device=parameter.device)
    groups = parameter.shape[0]
    energy = (
        parameter.grad.detach().float().reshape(groups, -1).square().sum(dim=1)
    )
    if groups == num_graphs:
        return energy
    graphs_per_group = num_graphs // groups
    return energy.div(graphs_per_group).repeat_interleave(graphs_per_group)


def _model_gradient_norms(scorer: ImplicitGraphScorer) -> tuple[Tensor, Tensor]:
    gate_energy = torch.zeros(
        scorer.num_layers, scorer.num_heads, device=scorer.device
    )
    for layer, gate in enumerate(scorer.gates):
        for parameter in (
            gate.q_proj.weight,
            gate.q_proj.bias,
            gate.k_proj.weight,
            gate.k_base,
            gate.b,
        ):
            gate_energy[layer] += _gradient_energy(parameter, scorer.num_heads)
        shared_energy = sum(
            parameter.grad.detach().float().square().sum()
            for module in (gate.q_norm, gate.k_norm)
            for parameter in module.parameters()
            if parameter.grad is not None
        )
        gate_energy[layer] += shared_energy / scorer.num_heads

    mixer_energy = torch.zeros(scorer.num_graphs, device=scorer.device)
    if scorer.mixer is not None:
        for parameter in scorer.mixer.parameters():
            mixer_energy += _mixer_gradient_energy(parameter, scorer.num_graphs)
    return gate_energy.flatten().sqrt(), mixer_energy.sqrt()


@dataclass(frozen=True)
class _PhaseResult:
    loss: Tensor
    optimizer_steps: int
    gate_gradient_norms: Tensor | None = None
    mixer_gradient_norms: Tensor | None = None


class GraphTrainer:
    """Run exact streamed mixer training for one whole context at a time."""

    def __init__(
        self,
        scorer: ImplicitGraphScorer,
        *,
        gate_optimizer=None,
        mixer_optimizer=None,
        gate_scheduler=None,
        mixer_scheduler=None,
        token_microbatch_size: int = 1000,
        graph_microbatch_size: str | int | None = None,
        subgraph_size: int | None = None,
        subgraphs_per_step: str | int = "max",
        shuffle_subgraphs: bool = False,
        timing: PhaseTiming | None = None,
    ) -> None:
        if (
            isinstance(token_microbatch_size, bool)
            or not isinstance(token_microbatch_size, int)
            or token_microbatch_size < 1
        ):
            raise ValueError("token microbatch size must be a positive integer")
        if subgraph_size is not None and (
            isinstance(subgraph_size, bool)
            or not isinstance(subgraph_size, int)
            or subgraph_size < 1
            or token_microbatch_size % subgraph_size
        ):
            raise ValueError(
                "subgraph size must be positive and divide the token microbatch size"
            )
        if subgraph_size is None and subgraphs_per_step != "max":
            raise ValueError("subgraphs per step requires a subgraph size")
        if subgraph_size is not None and subgraphs_per_step != "max":
            capacity = token_microbatch_size // subgraph_size
            if (
                isinstance(subgraphs_per_step, bool)
                or not isinstance(subgraphs_per_step, int)
                or subgraphs_per_step < 1
                or subgraphs_per_step % capacity
            ):
                raise ValueError(
                    "subgraphs per step must be positive and divisible by the "
                    "subgraphs in a token microbatch"
                )
        self.scorer = scorer
        self.gate_optimizer = gate_optimizer
        self.mixer_optimizer = mixer_optimizer
        self.gate_scheduler = gate_scheduler
        self.mixer_scheduler = mixer_scheduler
        self.token_microbatch_size = token_microbatch_size
        if scorer.scores_subgraphs_only and subgraph_size is None:
            raise ValueError(
                "the gps mixer trains on fixed-size subgraphs; set a subgraph "
                "size instead of training on whole contexts"
            )
        self.subgraph_size = subgraph_size
        self.subgraphs_per_step = subgraphs_per_step
        self.shuffle_subgraphs = shuffle_subgraphs
        self.graph_microbatch_size = (
            scorer.graph_microbatch_size if graph_microbatch_size is None else graph_microbatch_size
        )
        resolve_graph_microbatch_size(
            self.graph_microbatch_size, scorer.num_layers, scorer.num_heads
        )
        self.timing = timing

    @property
    def _device(self):
        return self.scorer.device

    @property
    def _dtype(self):
        return self.scorer.compute_dtype

    @property
    def _loss_dtype(self):
        return torch.float32 if self._dtype in {torch.float16, torch.bfloat16} else self._dtype

    def _timed(self, phase: str, operation: str):
        return nullcontext() if self.timing is None else self.timing.region(phase, operation)

    def _validate_example(self, example: TeacherExample) -> None:
        if len(example.hidden_by_layer) != self.scorer.num_layers:
            raise ValueError("teacher example layer count does not match scorer")
        if example.teacher_scores.shape[:3] != (
            self.scorer.num_layers,
            1,
            self.scorer.num_heads,
        ):
            raise ValueError("teacher score shape does not match scorer")
        if any(hidden.size(1) != self.scorer.hidden_dim for hidden in example.hidden_by_layer):
            raise ValueError("teacher hidden dimension does not match scorer")

    def _hidden(
        self, example, layer_ids: tuple[int, ...], positions: Tensor, offsets=None
    ) -> Tensor:
        offsets = (0,) * len(layer_ids) if offsets is None else offsets
        hidden = torch.stack(
            [
                example.hidden_by_layer[layer_id][positions + offset]
                for layer_id, offset in zip(layer_ids, offsets)
            ]
        )
        return hidden.to(device=self._device, dtype=self._dtype)

    def _targets(self, example, layer_ids, head_ids, positions, offsets=None) -> Tensor:
        offsets = (0,) * len(layer_ids) if offsets is None else offsets
        targets = torch.stack(
            [
                example.teacher_scores[layer_id, 0, head_id][positions + offset]
                for layer_id, head_id, offset in zip(layer_ids, head_ids, offsets)
            ]
        )
        return targets.to(device=self._device)

    @staticmethod
    def _bce_sum(scores: Tensor, targets: Tensor) -> Tensor:
        dtype = torch.float64 if scores.dtype == torch.float64 else torch.float32
        return F.binary_cross_entropy(
            scores.to(dtype), targets.to(dtype), reduction="sum"
        )

    def _token_chunks(self, token_count: int, *, shuffle: bool):
        positions = torch.randperm(token_count) if shuffle else torch.arange(token_count)
        for start in range(0, token_count, self.token_microbatch_size):
            yield positions[start : start + self.token_microbatch_size]

    @staticmethod
    def _step(optimizer, scheduler) -> None:
        optimizer.step()
        GraphTrainer._step_scheduler(scheduler)

    @staticmethod
    def _step_scheduler(scheduler) -> None:
        if scheduler is not None and not isinstance(
            scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
        ):
            scheduler.step()

    def _optimizer_batches(self, example: TeacherExample, *, shuffle: bool = False):
        if self.subgraph_size is None:
            yield ((None, example.sequence_length, 1),)
            return
        total = math.ceil(example.sequence_length / self.subgraph_size)
        limit = total if self.subgraphs_per_step == "max" else self.subgraphs_per_step
        subgraphs = [
            (start, token_count)
            for starts, token_count in subgraph_groups(
                example.sequence_length, self.subgraph_size, self.token_microbatch_size
            )
            for start in starts
        ]
        if shuffle:
            subgraphs = [subgraphs[index] for index in torch.randperm(total).tolist()]
        for first in range(0, total, limit):
            pending = []
            for start, token_count in subgraphs[first : first + limit]:
                if (
                    pending
                    and pending[-1][1] == token_count
                    and len(pending[-1][0]) * token_count < self.token_microbatch_size
                ):
                    starts, _, _ = pending[-1]
                    pending[-1] = (starts + (start,), token_count, total)
                else:
                    pending.append(((start,), token_count, total))
            yield tuple(pending)

    @staticmethod
    def _stacked_batch(batch, starts):
        if starts is None:
            return batch, None
        count = len(starts)
        return (
            GraphBatch(
                batch.graph_ids * count,
                batch.layer_ids * count,
                batch.head_ids * count,
            ),
            tuple(start for start in starts for _ in batch.graph_ids),
        )

    def _work_batches(self, example, batch):
        for optimizer_batch in self._optimizer_batches(example, shuffle=False):
            for starts, token_count, total in optimizer_batch:
                stacked, offsets = self._stacked_batch(batch, starts)
                yield stacked, offsets, token_count, total

    def _prepare(
        self,
        example: TeacherExample,
        batch,
        *,
        offsets=None,
        token_count=None,
        rnf_seed: int | None = None,
    ) -> PreparedImplicitGraph:
        token_count = example.sequence_length if token_count is None else token_count

        def chunks():
            for start in range(0, token_count, self.token_microbatch_size):
                stop = min(start + self.token_microbatch_size, token_count)
                positions = torch.arange(start, stop)
                yield start, self._hidden(
                    example, batch.layer_ids, positions, offsets
                )

        return self.scorer.mixer.prepare_from_chunks(
            chunks(),
            graph_ids=batch.graph_ids,
            token_count=token_count,
            token_microbatch_size=self.token_microbatch_size,
            rnf_seed=rnf_seed,
            offsets=offsets,
        )

    def _prepared_slice(self, prepared, positions: Tensor):
        return prepared.select_tokens(positions)

    def _score_from_normalized(
        self, hidden: Tensor, normalized: Tensor, batch
    ) -> Tensor:
        mixer = self.scorer.mixer
        alpha = mixer.alpha[list(batch.graph_ids)].to(normalized.dtype).view(-1, 1, 1)
        return self._score_from_correction(
            hidden, alpha * mixer.activated(normalized, batch.graph_ids), batch
        )

    def _score_from_transformed(
        self, hidden: Tensor, transformed: Tensor, batch
    ) -> Tensor:
        """Score from the graph-width pre-activation.

        Under the hidden coupling this is the GraNoLa affine output, finished by
        the out projection and alpha. Under the gate-space coupling it is any
        normalization's affine output, finished by the activation and the
        injection into the gate.
        """

        mixer = self.scorer.mixer
        if mixer.coupling == "gate-space":
            features = F.leaky_relu(transformed, negative_slope=mixer.leaky_relu_slope)
            return self._score_from_correction(
                hidden, mixer.injection(features, batch.graph_ids), batch
            )
        activated = mixer.projected_activation(transformed, batch.graph_ids)
        alpha = mixer.alpha[list(batch.graph_ids)].to(activated.dtype).view(-1, 1, 1)
        return self._score_from_correction(hidden, alpha * activated, batch)

    def _score_from_correction(self, hidden: Tensor, correction, batch) -> Tensor:
        # The gates hold master-dtype weights, while the trainer materializes
        # context hidden states in the compute dtype. `score_prepared` bridges
        # that gap for its own callers; this is the other way into the same
        # adapter, so it has to bridge it identically. Under the hidden
        # coupling the delta promoted the sum anyway, which is why only the
        # gate-space coupling, whose correction never touches the gate input,
        # made the difference visible.
        return self.scorer._gate_adapter.forward_batch(
            self.scorer.gates,
            batch.layer_ids,
            batch.head_ids,
            hidden.to(self.scorer.hidden_dtype),
            correction,
        )

    @staticmethod
    def _cache_prepared(prepared):
        return prepared.detached_to("cpu")

    def _cached_slice(self, prepared, positions: Tensor):
        return prepared.select_tokens(positions).detached_to(self._device)

    def train_gate_phase(self, example: TeacherExample) -> _PhaseResult:
        if self.subgraph_size is not None:
            raise ValueError("subgraph training requires joint mode")
        if self.gate_optimizer is None:
            raise ValueError("gate phase requires a gate optimizer")
        self._validate_example(example)
        for parameter in self.scorer.mixer.parameters():
            parameter.grad = None
        cached = []
        rnf_seed = (
            self.scorer.mixer.next_rnf_seed()
            if self.scorer.uses_granola
            else None
        )
        with _frozen(self.scorer.mixer.parameters()):
            with torch.no_grad():
                for batch in self.scorer.graph_batches(microbatch_size=self.graph_microbatch_size):
                    with self._timed("gate", "forward"):
                        cached.append(
                            (
                                batch,
                                self._cache_prepared(
                                    self._prepare(example, batch, rnf_seed=rnf_seed)
                                ),
                            )
                        )
            total_loss = torch.zeros((), device=self._device, dtype=self._loss_dtype)
            gradient_norms = torch.zeros(self.scorer.num_graphs, device=self._device)
            steps = 0
            for positions in self._token_chunks(example.sequence_length, shuffle=True):
                self.gate_optimizer.zero_grad(set_to_none=True)
                denominator = self.scorer.num_graphs * positions.numel()
                for batch, cached_prepared in cached:
                    with self._timed("gate", "forward"):
                        hidden = self._hidden(example, batch.layer_ids, positions)
                        prepared = self._cached_slice(cached_prepared, positions)
                        scores, _ = self.scorer.score_prepared(
                            hidden,
                            prepared,
                            layer_ids=batch.layer_ids,
                            head_ids=batch.head_ids,
                        )
                        numerator = self._bce_sum(
                            scores, self._targets(example, batch.layer_ids, batch.head_ids, positions)
                        )
                    with self._timed("gate", "backward"):
                        (numerator / denominator).backward()
                    total_loss += numerator.detach()
                gradient_norms += _model_gradient_norms(self.scorer)[0]
                self._step(self.gate_optimizer, self.gate_scheduler)
                steps += 1
        return _PhaseResult(
            loss=(total_loss / (self.scorer.num_graphs * example.sequence_length)).detach(),
            optimizer_steps=steps,
            gate_gradient_norms=gradient_norms / steps,
        )

    def _train_mixer_batch_autograd(
        self,
        example: TeacherExample,
        batch,
        *,
        phase: str,
        token_count: int,
        offsets=None,
        denominator=None,
    ) -> Tensor:
        """Score one subgraph batch and backpropagate it with ordinary autograd.

        The streamed replay below exists because the implicit mixer's state is a
        fixed-size Gram matrix, which lets a whole context be revisited in token
        slices. A GPS stack has no such summary: it keeps every token's
        activations. Its subgraphs are bounded, so autograd over the whole
        subgraph is both correct and affordable.
        """

        if denominator is None:
            denominator = self.scorer.num_graphs * token_count
        positions = torch.arange(token_count)
        with self._timed(phase, "forward"):
            hidden = self._hidden(example, batch.layer_ids, positions, offsets)
            prepared = self.scorer.prepare(
                hidden, batch.graph_ids, token_microbatch_size=token_count
            )
            scores, _ = self.scorer.score_prepared(
                hidden,
                prepared,
                layer_ids=batch.layer_ids,
                head_ids=batch.head_ids,
            )
            numerator = self._bce_sum(
                scores,
                self._targets(
                    example, batch.layer_ids, batch.head_ids, positions, offsets
                ),
            )
        with self._timed(phase, "backward"):
            (numerator / denominator).backward()
        return numerator.detach()

    def _train_mixer_batch(
        self,
        example: TeacherExample,
        batch,
        prepared: PreparedImplicitGraph,
        *,
        joint: bool,
        phase: str,
        offsets=None,
        denominator=None,
    ) -> Tensor:
        """Backpropagate the selected normalization without retaining full P."""

        mixer = self.scorer.mixer
        if mixer.normalization == "granola" or mixer.coupling == "gate-space":
            return self._train_graph_width_batch(
                example,
                batch,
                prepared,
                phase=phase,
                offsets=offsets,
                denominator=denominator,
            )
        graph_count, token_count, graph_dim = prepared.y1.shape
        hidden_dim = self.scorer.hidden_dim
        work_dtype = torch.float64 if prepared.y1.dtype == torch.float64 else torch.float32
        direct_y1_gradient = torch.zeros(
            (graph_count, token_count, graph_dim),
            device=self._device,
            dtype=work_dtype,
        )
        kernel_gradient = torch.zeros(
            (graph_count, graph_dim, hidden_dim),
            device=self._device,
            dtype=work_dtype,
        )
        total_numerator = torch.zeros((), device=self._device, dtype=self._loss_dtype)
        if denominator is None:
            denominator = self.scorer.num_graphs * token_count
        # With self loops the pre-activation gains `lambda (Y2 W^T)`, a term the
        # kernel does not carry, so its gradient is collected separately.
        direct_y2_gradient = (
            None if mixer.self_loop is None else torch.zeros_like(direct_y1_gradient)
        )

        def absorb_raw_gradient(positions: Tensor, gradient: Tensor) -> None:
            index = positions.to(self._device)
            values = gradient.detach().to(work_dtype)
            y1 = prepared.y1.index_select(1, index).to(work_dtype)
            direct_y1_gradient.index_add_(
                1,
                index,
                torch.bmm(values, prepared.kernel.to(work_dtype).transpose(1, 2)),
            )
            kernel_gradient.add_(torch.bmm(y1.transpose(1, 2), values))
            if direct_y2_gradient is not None:
                # Autograd through a live copy of the self-loop term gives the
                # weight, the out projection rows and the y2 chunk their
                # gradients; the chunk's is kept for the projection pass.
                y2_chunk = (
                    prepared.y2.index_select(1, index).to(work_dtype).detach().requires_grad_(True)
                )
                with self._timed(phase, "forward"):
                    weight = (
                        _select_graph_rows(mixer.self_loop, batch.graph_ids)
                        .to(work_dtype)
                        .view(-1, 1, 1)
                    )
                    out_weight = _select_graph_rows(
                        mixer.out_proj.weight, batch.graph_ids
                    ).to(work_dtype)
                    term = weight * torch.bmm(y2_chunk, out_weight.transpose(1, 2))
                with self._timed(phase, "backward"):
                    torch.autograd.backward(term, values)
                direct_y2_gradient.index_copy_(1, index, y2_chunk.grad.detach())

        batchnorm_chunks: list[tuple[Tensor, Tensor]] = []
        sum_h = sum_hx = None
        if mixer.normalization == "batchnorm":
            sum_h = torch.zeros(
                (graph_count, hidden_dim), device=self._device, dtype=work_dtype
            )
            sum_hx = torch.zeros_like(sum_h)

        for positions in self._token_chunks(token_count, shuffle=False):
            with self._timed(phase, "forward"):
                sliced = self._prepared_slice(prepared, positions)
                raw = mixer._raw_with_self_loop(
                    sliced.y1, sliced.y2, sliced.kernel, batch.graph_ids
                )
                hidden = self._hidden(example, batch.layer_ids, positions, offsets)
                if mixer.normalization == "batchnorm":
                    normalized = mixer.normalized(raw, sliced).detach().requires_grad_(True)
                    scores = self._score_from_normalized(hidden, normalized, batch)
                else:
                    raw_proxy = raw.detach().requires_grad_(True)
                    normalized = mixer.normalized(raw_proxy, sliced)
                    scores = self._score_from_normalized(hidden, normalized, batch)
                numerator = self._bce_sum(
                    scores,
                    self._targets(
                        example, batch.layer_ids, batch.head_ids, positions, offsets
                    ),
                )
            with self._timed(phase, "backward"):
                (numerator / denominator).backward()
            if mixer.normalization == "batchnorm":
                assert sum_h is not None and sum_hx is not None
                gradient = normalized.grad.detach()
                sum_h += gradient.sum(dim=1)
                sum_hx += (gradient * normalized.detach()).sum(dim=1)
                batchnorm_chunks.append((positions, gradient))
            else:
                absorb_raw_gradient(positions, raw_proxy.grad)
            total_numerator += numerator.detach()

        if mixer.normalization == "batchnorm":
            if not isinstance(prepared.norm, ContextNormStats):
                raise ValueError("prepared graph is missing BatchNorm statistics")
            mean_h = sum_h / token_count
            mean_hx = sum_hx / token_count
            for positions, gradient in batchnorm_chunks:
                with self._timed(phase, "forward"):
                    sliced = self._prepared_slice(prepared, positions)
                    raw = mixer._raw_with_self_loop(
                        sliced.y1, sliced.y2, sliced.kernel, batch.graph_ids
                    )
                    normalized = mixer.normalized(raw, prepared)
                    raw_gradient = prepared.norm.invstd.unsqueeze(1) * (
                        gradient
                        - mean_h.unsqueeze(1)
                        - normalized * mean_hx.unsqueeze(1)
                    )
                absorb_raw_gradient(positions, raw_gradient)

        gram_proxy = prepared.gram.detach().requires_grad_(True)
        with self._timed(phase, "forward"):
            live_kernel = mixer._kernel(gram_proxy, batch.graph_ids)
        with self._timed(phase, "backward"):
            torch.autograd.backward(live_kernel, kernel_gradient)
        gram_gradient = gram_proxy.grad.detach()
        self._absorb_projection_gradients(
            example,
            batch,
            prepared,
            phase=phase,
            offsets=offsets,
            direct_y1_gradient=direct_y1_gradient,
            direct_y2_gradient=direct_y2_gradient,
            gram_gradient=gram_gradient,
            work_dtype=work_dtype,
        )
        return total_numerator

    def _train_graph_width_batch(
        self,
        example: TeacherExample,
        batch,
        prepared: PreparedImplicitGraph,
        *,
        phase: str,
        offsets=None,
        denominator=None,
    ) -> Tensor:
        """Backpropagate with one live graph-width autograd graph.

        Everything upstream of the pre-activation is graph width, so the whole
        subgraph from the two projections down to the affine output fits in
        memory and ordinary autograd handles it. Only the tail (activation,
        out projection or injection, gate, loss) is streamed per token chunk,
        and the single tensor crossing that boundary is the affine output, so
        its gradient is all the bookkeeping this needs.

        The hidden coupling takes this route for GraNoLa, whose affine runs at
        graph width. The gate-space coupling takes it for every normalization,
        because nothing in it ever reaches hidden width.
        """

        mixer = self.scorer.mixer
        graph_count, token_count, _ = prepared.y1.shape
        work_dtype = torch.float64 if prepared.y1.dtype == torch.float64 else torch.float32
        granola = mixer.normalization == "granola"
        if granola and not isinstance(prepared.norm, _GranolaNormState):
            raise ValueError("prepared graph is missing GraNoLa state")
        if prepared.y2 is None:
            raise ValueError("graph-width training requires the retained message features")
        total_numerator = torch.zeros((), device=self._device, dtype=self._loss_dtype)
        if denominator is None:
            denominator = self.scorer.num_graphs * token_count

        y1 = prepared.y1.detach().to(work_dtype).requires_grad_(True)
        y2 = prepared.y2.detach().to(work_dtype).requires_grad_(True)
        scale = token_count if mixer.gram_normalization == "token-count" else 1
        group_ids = mixer.normalization_group_ids(batch.graph_ids)
        with self._timed(phase, "forward"):
            gram = torch.bmm(y1.transpose(1, 2), y2) / scale
            if granola:
                # The message is formed after the GNN on purpose: autograd
                # accumulates the y1/y2 gradients in graph order, and keeping
                # the original order keeps GraNoLa's gradients bit-identical.
                hidden_state = mixer.granola_gnn(
                    y1, y2, prepared.norm.rnf.to(work_dtype), group_ids, scale=scale
                )
                gamma, beta = mixer.granola_affine(
                    mixer.granola_readout(hidden_state), batch.graph_ids
                )
                normalized = mixer.granola_normalized(
                    mixer.message(y1, y2, gram, batch.graph_ids)
                )
                transformed = gamma * normalized + beta
            elif mixer.normalization == "batchnorm":
                message = mixer.message(y1, y2, gram, batch.graph_ids)
                gamma = _select_graph_rows(mixer.gamma, group_ids).to(work_dtype).unsqueeze(1)
                beta = _select_graph_rows(mixer.beta, group_ids).to(work_dtype).unsqueeze(1)
                transformed = gamma * mixer.context_normalized(message) + beta
            else:
                transformed = mixer.message(y1, y2, gram, batch.graph_ids)

        # Detaching here is load-bearing: slicing the live tensor would make
        # every chunk's backward walk the whole GNN again.
        values = transformed.detach()
        transformed_gradient = torch.zeros_like(values)

        for positions in self._token_chunks(token_count, shuffle=False):
            index = positions.to(self._device)
            with self._timed(phase, "forward"):
                chunk = values.index_select(1, index).requires_grad_(True)
                hidden = self._hidden(example, batch.layer_ids, positions, offsets)
                scores = self._score_from_transformed(hidden, chunk, batch)
                numerator = self._bce_sum(
                    scores,
                    self._targets(
                        example, batch.layer_ids, batch.head_ids, positions, offsets
                    ),
                )
            with self._timed(phase, "backward"):
                (numerator / denominator).backward()
            transformed_gradient.index_copy_(
                1, index, chunk.grad.detach().to(work_dtype)
            )
            total_numerator += numerator.detach()

        with self._timed(phase, "backward"):
            torch.autograd.backward(
                transformed, transformed_gradient.to(transformed.dtype)
            )
        self._absorb_projection_gradients(
            example,
            batch,
            prepared,
            phase=phase,
            offsets=offsets,
            direct_y1_gradient=y1.grad.detach().to(work_dtype),
            direct_y2_gradient=y2.grad.detach().to(work_dtype),
            gram_gradient=None,
            work_dtype=work_dtype,
        )
        return total_numerator

    def _absorb_projection_gradients(
        self,
        example: TeacherExample,
        batch,
        prepared: PreparedImplicitGraph,
        *,
        phase: str,
        offsets,
        direct_y1_gradient: Tensor,
        direct_y2_gradient: Tensor | None,
        gram_gradient: Tensor | None,
        work_dtype: torch.dtype,
    ) -> None:
        """Push the Y1/Y2 gradients through in_proj, one token chunk at a time.

        The context hidden states are hidden width and stream in from the host,
        so this stays chunked whichever normalization produced the gradients.

        This needs the whole context. It walks chunk positions and uses them
        both to index the gradient buffers and to fetch the matching context
        hidden states, which only line up when the prepared state still covers
        every token. A sliced state keeps its original token_count while its
        projections shrink, so it would pair each gradient with the wrong
        token; refuse it rather than train on that quietly.
        """

        mixer = self.scorer.mixer
        graph_dim = prepared.y1.size(-1)
        token_count = prepared.y1.size(1)
        if token_count != prepared.token_count:
            raise ValueError(
                "mixer gradients need the complete context, not a token slice"
            )
        scale = token_count if mixer.gram_normalization == "token-count" else 1
        for positions in self._token_chunks(token_count, shuffle=False):
            index = positions.to(self._device)
            with self._timed(phase, "forward"):
                hidden = self._hidden(example, batch.layer_ids, positions, offsets)
                packed = mixer.in_proj(hidden, batch.graph_ids)
                y1_gradient = direct_y1_gradient.index_select(1, index)
                if direct_y2_gradient is None:
                    y2_gradient = torch.zeros_like(y1_gradient)
                else:
                    y2_gradient = direct_y2_gradient.index_select(1, index)
                if gram_gradient is not None:
                    # The GraNoLa path differentiates the Gram matrix directly,
                    # so only the staged branches add its contribution here.
                    _, y2_live = packed.split(graph_dim, dim=-1)
                    y1_gradient = y1_gradient + torch.bmm(
                        y2_live.to(work_dtype), gram_gradient.transpose(1, 2)
                    ) / scale
                    y2_gradient = y2_gradient + torch.bmm(
                        prepared.y1.index_select(1, index).to(work_dtype),
                        gram_gradient,
                    ) / scale
                packed_gradient = torch.cat(
                    (y1_gradient, y2_gradient), dim=-1
                ).to(packed.dtype)
            with self._timed(phase, "backward"):
                torch.autograd.backward(packed, packed_gradient)

    def train_mixer_phase(
        self, example: TeacherExample, *, joint: bool = False
    ) -> _PhaseResult:
        if self.subgraph_size is not None and not joint:
            raise ValueError("subgraph training requires joint mode")
        if self.mixer_optimizer is None:
            raise ValueError("graph phase requires a mixer optimizer")
        self._validate_example(example)
        phase = "joint" if joint else "graph"
        if not joint:
            for parameter in self.scorer.gates.parameters():
                parameter.grad = None
        gate_context = nullcontext() if joint else _frozen(self.scorer.gates.parameters())
        total_loss = torch.zeros((), device=self._device, dtype=self._loss_dtype)
        gate_gradient_norms = torch.zeros(self.scorer.num_graphs, device=self._device)
        mixer_gradient_norms = torch.zeros_like(gate_gradient_norms)
        steps = 0
        rnf_seed = (
            self.scorer.mixer.next_rnf_seed()
            if self.scorer.uses_granola
            else None
        )
        with gate_context:
            for optimizer_batch in self._optimizer_batches(
                example, shuffle=self.shuffle_subgraphs
            ):
                self.mixer_optimizer.zero_grad(set_to_none=True)
                if joint and self.gate_optimizer is not None:
                    self.gate_optimizer.zero_grad(set_to_none=True)
                step_subgraphs = sum(
                    1 if starts is None else len(starts)
                    for starts, _, _ in optimizer_batch
                )
                for base_batch in self.scorer.graph_batches(
                    microbatch_size=self.graph_microbatch_size
                ):
                    for starts, token_count, total_subgraphs in optimizer_batch:
                        batch, offsets = self._stacked_batch(base_batch, starts)
                        denominator = (
                            self.scorer.num_graphs * step_subgraphs * token_count
                        )
                        if self.scorer.scores_subgraphs_only:
                            numerator = self._train_mixer_batch_autograd(
                                example,
                                batch,
                                phase=phase,
                                token_count=token_count,
                                offsets=offsets,
                                denominator=denominator,
                            )
                        else:
                            with torch.no_grad():
                                with self._timed(phase, "forward"):
                                    prepared = self._prepare(
                                        example,
                                        batch,
                                        offsets=offsets,
                                        token_count=token_count,
                                        rnf_seed=rnf_seed,
                                    )
                            numerator = self._train_mixer_batch(
                                example,
                                batch,
                                prepared,
                                joint=joint,
                                phase=phase,
                                offsets=offsets,
                                denominator=denominator,
                            )
                        total_loss += numerator / (
                            self.scorer.num_graphs
                            * total_subgraphs
                            * token_count
                        )
                gate_norms, mixer_norms = _model_gradient_norms(self.scorer)
                gate_gradient_norms += gate_norms
                mixer_gradient_norms += mixer_norms
                self.mixer_optimizer.step()
                self.scorer.mixer.on_optimizer_step()
                if joint and self.gate_optimizer is not None:
                    self.gate_optimizer.step()
                steps += 1
        self._step_scheduler(self.mixer_scheduler)
        if joint and self.gate_optimizer is not None:
            self._step_scheduler(self.gate_scheduler)
        return _PhaseResult(
            loss=total_loss.detach(),
            optimizer_steps=steps,
            gate_gradient_norms=gate_gradient_norms / steps if joint else None,
            mixer_gradient_norms=mixer_gradient_norms / steps,
        )

    def evaluate_context(self, example: TeacherExample) -> _PhaseResult:
        self._validate_example(example)
        total_loss = torch.zeros((), device=self._device, dtype=self._loss_dtype)
        rnf_seed = (
            derive_evaluation_rnf_seed(
                self.scorer.mixer.normalization_seed,
                example.dataset_name,
                example.dataset_index,
            )
            if self.scorer.uses_granola
            else None
        )
        with torch.no_grad():
            for base_batch in self.scorer.graph_batches(
                microbatch_size=self.graph_microbatch_size
            ):
                for batch, offsets, token_count, total_subgraphs in self._work_batches(
                    example, base_batch
                ):
                    with self._timed("graph", "forward"):
                        prepared = self._prepare(
                            example,
                            batch,
                            offsets=offsets,
                            token_count=token_count,
                            rnf_seed=rnf_seed,
                        )
                    for positions in self._token_chunks(token_count, shuffle=False):
                        with self._timed("graph", "forward"):
                            hidden = self._hidden(
                                example, batch.layer_ids, positions, offsets
                            )
                            scores, _ = self.scorer.score_prepared(
                                hidden,
                                self._prepared_slice(prepared, positions),
                                layer_ids=batch.layer_ids,
                                head_ids=batch.head_ids,
                            )
                            numerator = self._bce_sum(
                                scores,
                                self._targets(
                                    example,
                                    batch.layer_ids,
                                    batch.head_ids,
                                    positions,
                                    offsets,
                                ),
                            )
                            total_loss += (
                                numerator
                                if self.subgraph_size is None
                                else numerator
                                / (self.scorer.num_graphs * total_subgraphs * token_count)
                            )
        return _PhaseResult(
            loss=(
                total_loss / (self.scorer.num_graphs * example.sequence_length)
                if self.subgraph_size is None
                else total_loss
            ).detach(),
            optimizer_steps=0,
        )

    def step_validation(self, loss: float) -> None:
        for scheduler in (self.gate_scheduler, self.mixer_scheduler):
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(loss)

    def train_context(self, example: TeacherExample, *, mode: str = "joint"):
        if mode not in {"gate", "graph", "two_phase", "joint"}:
            raise ValueError("training mode must be gate, graph, two_phase, or joint")
        gate_result = None
        graph_result = None
        if mode in {"gate", "two_phase"} and self.gate_optimizer is not None:
            gate_result = self.train_gate_phase(example)
        if mode in {"graph", "two_phase"}:
            graph_result = self.train_mixer_phase(example)
        elif mode == "joint":
            graph_result = self.train_mixer_phase(example, joint=True)
        zeros = torch.zeros(self.scorer.num_graphs, device=self._device)
        gate_gradient_norms = (
            gate_result.gate_gradient_norms
            if gate_result is not None
            else graph_result.gate_gradient_norms if graph_result is not None else None
        )
        mixer_gradient_norms = (
            graph_result.mixer_gradient_norms if graph_result is not None else None
        )
        gate_gradient_norms = zeros if gate_gradient_norms is None else gate_gradient_norms
        mixer_gradient_norms = zeros if mixer_gradient_norms is None else mixer_gradient_norms
        return {
            "gate_loss": None if gate_result is None else gate_result.loss,
            "graph_loss": None if graph_result is None or mode == "joint" else graph_result.loss,
            "joint_loss": None if graph_result is None or mode != "joint" else graph_result.loss,
            "gate_steps": (
                gate_result.optimizer_steps
                if gate_result is not None
                else (
                    graph_result.optimizer_steps
                    if mode == "joint" and self.gate_optimizer is not None
                    else 0
                )
            ),
            "mixer_steps": 0 if graph_result is None else graph_result.optimizer_steps,
            "gradient_norm": torch.sqrt(
                gate_gradient_norms.square() + mixer_gradient_norms.square()
            ).mean(),
            "gate_gradient_norm": gate_gradient_norms.mean(),
            "mixer_gradient_norm": mixer_gradient_norms.mean(),
        }
