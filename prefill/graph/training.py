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
    ImplicitGraphScorer,
    PreparedImplicitGraph,
    compute_dtype_name,
    parse_compute_dtype,
    resolve_graph_microbatch_size,
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
    scheduler_class = _scheduler_class(name)
    parameter = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.AdamW([parameter], lr=1e-3)
    try:
        scheduler_class(optimizer, **parsed)
    except Exception as error:
        raise ValueError(f"invalid {name} scheduler arguments: {error}") from error
    return SchedulerSpec(name, parsed)


def build_scheduler(optimizer, spec: SchedulerSpec | None):
    return None if spec is None else _scheduler_class(spec.name)(optimizer, **spec.kwargs)


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
    gate_frozen: bool = False,
    mixer_frozen: bool = False,
):
    """Return disjoint gate and mixer AdamW optimizers."""

    gate_lr, mixer_lr = _valid_lr(gate_lr), _valid_lr(mixer_lr)
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("weight decay must be finite and non-negative")
    gate_parameters = list(scorer.gates.parameters())
    mixer = scorer.mixer
    decay_parameters = [mixer.in_proj.weight, mixer.out_proj.weight]
    no_decay_parameters = [mixer.alpha, mixer.gamma, mixer.beta]
    for parameter in gate_parameters:
        parameter.requires_grad_(not gate_frozen)
    for parameter in (*decay_parameters, *no_decay_parameters):
        parameter.requires_grad_(not mixer_frozen)
    gate_optimizer = None
    mixer_optimizer = None
    if not gate_frozen:
        gate_optimizer = torch.optim.AdamW(
            gate_parameters, lr=gate_lr, weight_decay=weight_decay
        )
    if not mixer_frozen:
        mixer_optimizer = torch.optim.AdamW(
            [
                {"params": decay_parameters, "weight_decay": weight_decay},
                {"params": no_decay_parameters, "weight_decay": 0.0},
            ],
            lr=mixer_lr,
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
    payload = torch.load(path, map_location="cpu", weights_only=False)
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


@dataclass(frozen=True)
class _PhaseResult:
    loss: Tensor
    optimizer_steps: int


class GraphTrainer:
    """Run exact streamed BatchNorm training for one whole context at a time."""

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
        timing: PhaseTiming | None = None,
    ) -> None:
        if (
            isinstance(token_microbatch_size, bool)
            or not isinstance(token_microbatch_size, int)
            or token_microbatch_size < 1
        ):
            raise ValueError("token microbatch size must be a positive integer")
        self.scorer = scorer
        self.gate_optimizer = gate_optimizer
        self.mixer_optimizer = mixer_optimizer
        self.gate_scheduler = gate_scheduler
        self.mixer_scheduler = mixer_scheduler
        self.token_microbatch_size = token_microbatch_size
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

    def _hidden(self, example, layer_ids: tuple[int, ...], positions: Tensor) -> Tensor:
        hidden = torch.stack(
            [example.hidden_by_layer[layer_id][positions] for layer_id in layer_ids]
        )
        return hidden.to(device=self._device, dtype=self._dtype)

    def _targets(self, example, layer_ids, head_ids, positions) -> Tensor:
        targets = torch.stack(
            [
                example.teacher_scores[layer_id, 0, head_id]
                for layer_id, head_id in zip(layer_ids, head_ids)
            ]
        )[:, positions]
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
        if scheduler is not None and not isinstance(
            scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
        ):
            scheduler.step()

    def _prepare(self, example: TeacherExample, batch) -> PreparedImplicitGraph:
        token_count = example.sequence_length

        def chunks():
            for start in range(0, token_count, self.token_microbatch_size):
                stop = min(start + self.token_microbatch_size, token_count)
                positions = torch.arange(start, stop)
                yield start, self._hidden(example, batch.layer_ids, positions)

        return self.scorer.mixer.prepare_from_chunks(
            chunks(),
            graph_ids=batch.graph_ids,
            token_count=token_count,
            token_microbatch_size=self.token_microbatch_size,
        )

    def _prepared_slice(
        self, prepared: PreparedImplicitGraph, positions: Tensor
    ) -> PreparedImplicitGraph:
        index = positions.to(prepared.y1.device)
        return PreparedImplicitGraph(
            prepared.graph_ids,
            prepared.y1.index_select(1, index),
            prepared.gram,
            prepared.kernel,
            prepared.norm,
            prepared.token_count,
        )

    def _score_from_normalized(self, hidden: Tensor, normalized: Tensor, batch) -> Tensor:
        mixer = self.scorer.mixer
        gamma = mixer.gamma[list(batch.graph_ids)].to(normalized.dtype).unsqueeze(1)
        beta = mixer.beta[list(batch.graph_ids)].to(normalized.dtype).unsqueeze(1)
        alpha = mixer.alpha[list(batch.graph_ids)].to(normalized.dtype).view(-1, 1, 1)
        delta = alpha * (gamma * normalized + beta)
        return torch.stack(
            [
                self.scorer._gate_adapter(
                    self.scorer.gates[layer_id],
                    head_id,
                    hidden[local_graph],
                    delta[local_graph],
                )
                for local_graph, (layer_id, head_id) in enumerate(
                    zip(batch.layer_ids, batch.head_ids)
                )
            ]
        )

    @staticmethod
    def _cache_prepared(prepared: PreparedImplicitGraph) -> PreparedImplicitGraph:
        return PreparedImplicitGraph(
            prepared.graph_ids,
            prepared.y1.detach().to("cpu"),
            prepared.gram.detach().to("cpu"),
            prepared.kernel.detach().to("cpu"),
            type(prepared.norm)(
                prepared.norm.mean.detach().to("cpu"),
                prepared.norm.invstd.detach().to("cpu"),
            ),
            prepared.token_count,
        )

    def _cached_slice(
        self, prepared: PreparedImplicitGraph, positions: Tensor
    ) -> PreparedImplicitGraph:
        return PreparedImplicitGraph(
            prepared.graph_ids,
            prepared.y1[:, positions].to(self._device),
            prepared.gram.to(self._device),
            prepared.kernel.to(self._device),
            type(prepared.norm)(
                prepared.norm.mean.to(self._device), prepared.norm.invstd.to(self._device)
            ),
            prepared.token_count,
        )

    def train_gate_phase(self, example: TeacherExample) -> _PhaseResult:
        if self.gate_optimizer is None:
            raise ValueError("gate phase requires a gate optimizer")
        self._validate_example(example)
        for parameter in self.scorer.mixer.parameters():
            parameter.grad = None
        cached = []
        with _frozen(self.scorer.mixer.parameters()):
            with torch.no_grad():
                for batch in self.scorer.graph_batches(microbatch_size=self.graph_microbatch_size):
                    with self._timed("gate", "forward"):
                        cached.append((batch, self._cache_prepared(self._prepare(example, batch))))
            total_loss = torch.zeros((), device=self._device, dtype=self._loss_dtype)
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
                self._step(self.gate_optimizer, self.gate_scheduler)
                steps += 1
        return _PhaseResult(
            loss=(total_loss / (self.scorer.num_graphs * example.sequence_length)).detach(),
            optimizer_steps=steps,
        )

    def _train_mixer_batch(
        self,
        example: TeacherExample,
        batch,
        prepared: PreparedImplicitGraph,
        *,
        joint: bool,
        phase: str,
    ) -> Tensor:
        """Backpropagate current-context BN exactly in two streamed loss passes."""

        mixer = self.scorer.mixer
        graph_count, token_count, graph_dim = prepared.y1.shape
        hidden_dim = self.scorer.hidden_dim
        work_dtype = torch.float64 if prepared.y1.dtype == torch.float64 else torch.float32
        direct_y1_gradient = torch.zeros(
            (graph_count, token_count, graph_dim),
            device=self._device,
            dtype=work_dtype,
        )
        sum_h = torch.zeros((graph_count, hidden_dim), device=self._device, dtype=work_dtype)
        sum_hx = torch.zeros_like(sum_h)
        total_numerator = torch.zeros((), device=self._device, dtype=self._loss_dtype)
        denominator = self.scorer.num_graphs * token_count
        saved_chunks: list[tuple[Tensor, Tensor]] = []

        for positions in self._token_chunks(token_count, shuffle=False):
            with self._timed(phase, "forward"):
                sliced = self._prepared_slice(prepared, positions)
                raw = mixer._raw(sliced.y1, sliced.kernel)
                normalized = mixer.normalized(raw, sliced).detach().requires_grad_(True)
                hidden = self._hidden(example, batch.layer_ids, positions)
                scores = self._score_from_normalized(hidden, normalized, batch)
                numerator = self._bce_sum(
                    scores, self._targets(example, batch.layer_ids, batch.head_ids, positions)
                )
            with self._timed(phase, "backward"):
                (numerator / denominator).backward()
            h = normalized.grad.detach()
            sum_h += h.sum(dim=1)
            sum_hx += (h * normalized.detach()).sum(dim=1)
            saved_chunks.append((positions, h))
            total_numerator += numerator.detach()

        mean_h = sum_h / token_count
        mean_hx = sum_hx / token_count
        kernel_proxy = prepared.kernel.detach().requires_grad_(True)
        for positions, h in saved_chunks:
            with self._timed(phase, "forward"):
                y1_proxy = self._prepared_slice(prepared, positions).y1.detach().requires_grad_(True)
                raw = mixer._raw(y1_proxy, kernel_proxy)
                normalized = mixer.normalized(raw, prepared)
                raw_gradient = prepared.norm.invstd.unsqueeze(1) * (
                    h - mean_h.unsqueeze(1) - normalized * mean_hx.unsqueeze(1)
                )
            with self._timed(phase, "backward"):
                torch.autograd.backward(raw, raw_gradient)
            direct_y1_gradient[:, positions.to(self._device)] = y1_proxy.grad.detach()

        gram_proxy = prepared.gram.detach().requires_grad_(True)
        with self._timed(phase, "forward"):
            live_kernel = mixer._kernel(gram_proxy, batch.graph_ids)
        with self._timed(phase, "backward"):
            torch.autograd.backward(live_kernel, kernel_proxy.grad)
        gram_gradient = gram_proxy.grad.detach()
        scale = token_count if mixer.gram_normalization == "token-count" else 1

        for positions in self._token_chunks(token_count, shuffle=False):
            with self._timed(phase, "forward"):
                hidden = self._hidden(example, batch.layer_ids, positions)
                packed = mixer.in_proj(hidden, batch.graph_ids)
                y1_live, y2_live = packed.split(graph_dim, dim=-1)
                index = positions.to(self._device)
                y1_gradient = direct_y1_gradient[:, index] + torch.bmm(
                    y2_live.to(work_dtype), gram_gradient.transpose(1, 2)
                ) / scale
                y2_gradient = torch.bmm(
                    prepared.y1[:, index].to(work_dtype), gram_gradient
                ) / scale
                packed_gradient = torch.cat((y1_gradient, y2_gradient), dim=-1).to(
                    packed.dtype
                )
            with self._timed(phase, "backward"):
                torch.autograd.backward(packed, packed_gradient)
        return total_numerator

    def train_mixer_phase(
        self, example: TeacherExample, *, joint: bool = False
    ) -> _PhaseResult:
        if self.mixer_optimizer is None:
            raise ValueError("graph phase requires a mixer optimizer")
        self._validate_example(example)
        phase = "joint" if joint else "graph"
        self.mixer_optimizer.zero_grad(set_to_none=True)
        if joint and self.gate_optimizer is not None:
            self.gate_optimizer.zero_grad(set_to_none=True)
        if not joint:
            for parameter in self.scorer.gates.parameters():
                parameter.grad = None
        gate_context = nullcontext() if joint else _frozen(self.scorer.gates.parameters())
        total_numerator = torch.zeros((), device=self._device, dtype=self._loss_dtype)
        with gate_context:
            for batch in self.scorer.graph_batches(microbatch_size=self.graph_microbatch_size):
                with torch.no_grad():
                    with self._timed(phase, "forward"):
                        prepared = self._prepare(example, batch)
                total_numerator += self._train_mixer_batch(
                    example, batch, prepared, joint=joint, phase=phase
                )
        self._step(self.mixer_optimizer, self.mixer_scheduler)
        if joint and self.gate_optimizer is not None:
            self._step(self.gate_optimizer, self.gate_scheduler)
        return _PhaseResult(
            loss=(total_numerator / (self.scorer.num_graphs * example.sequence_length)).detach(),
            optimizer_steps=1,
        )

    def evaluate_context(self, example: TeacherExample) -> _PhaseResult:
        self._validate_example(example)
        total_loss = torch.zeros((), device=self._device, dtype=self._loss_dtype)
        with torch.no_grad():
            for batch in self.scorer.graph_batches(microbatch_size=self.graph_microbatch_size):
                with self._timed("graph", "forward"):
                    prepared = self._prepare(example, batch)
                for positions in self._token_chunks(example.sequence_length, shuffle=False):
                    with self._timed("graph", "forward"):
                        hidden = self._hidden(example, batch.layer_ids, positions)
                        scores, _ = self.scorer.score_prepared(
                            hidden,
                            self._prepared_slice(prepared, positions),
                            layer_ids=batch.layer_ids,
                            head_ids=batch.head_ids,
                        )
                        total_loss += self._bce_sum(
                            scores, self._targets(example, batch.layer_ids, batch.head_ids, positions)
                        )
        return _PhaseResult(
            loss=(total_loss / (self.scorer.num_graphs * example.sequence_length)).detach(),
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
        return {
            "gate_loss": None if gate_result is None else gate_result.loss,
            "graph_loss": None if graph_result is None or mode == "joint" else graph_result.loss,
            "joint_loss": None if graph_result is None or mode != "joint" else graph_result.loss,
            "gate_steps": (
                gate_result.optimizer_steps
                if gate_result is not None
                else int(mode == "joint" and self.gate_optimizer is not None)
            ),
            "mixer_steps": 0 if graph_result is None else graph_result.optimizer_steps,
        }
