"""Training primitives for whole-context graph scoring."""

from __future__ import annotations

import copy
import json
import math
import random
from contextlib import contextmanager
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from .model import GraphScorer, resolve_graph_microbatch_size


def _normal_cpu_tensor(tensor: Tensor) -> Tensor:
    return tensor.detach().to("cpu").clone()


def _normal_hidden_tensor(tensor: Tensor) -> Tensor:
    tensor = _normal_cpu_tensor(tensor)
    if tensor.ndim == 3 and tensor.size(0) == 1:
        tensor = tensor[0]
    return tensor


@dataclass(frozen=True)
class TeacherExample:
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
        if self.sequence_length < 1:
            raise ValueError("sequence length must be positive")
        if not hidden or any(tensor.ndim != 2 for tensor in hidden):
            raise ValueError("hidden tensors must have shape [tokens, hidden_dim]")
        if any(tensor.size(0) != self.sequence_length for tensor in hidden):
            raise ValueError("hidden tensor sequence length does not match example")
        if token_ids.ndim not in {1, 2} or token_ids.size(-1) != self.sequence_length:
            raise ValueError("token ID sequence length does not match example")
        if teacher_scores.ndim != 4 or teacher_scores.size(-1) != self.sequence_length:
            raise ValueError("teacher score sequence length does not match example")
        if teacher_scores.size(0) != len(hidden):
            raise ValueError("teacher score layers do not match hidden tensors")
        object.__setattr__(self, "hidden_by_layer", hidden)
        object.__setattr__(self, "token_ids", token_ids)
        object.__setattr__(self, "teacher_scores", teacher_scores)
        object.__setattr__(self, "prefix_ids", prefix_ids)


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
    """Parse and instantiate-check a scheduler without loading the model."""

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
    if spec is None:
        return None
    return _scheduler_class(spec.name)(optimizer, **spec.kwargs)


def _valid_lr(value: float | None) -> float | None:
    if value is not None and (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError("learning rates must be finite and positive")
    return value


def resolve_joint_settings(
    gate_lr: float | None,
    graph_lr: float | None,
    gate_scheduler: SchedulerSpec | None,
    graph_scheduler: SchedulerSpec | None,
    *,
    gate_frozen: bool = False,
) -> tuple[float, float, SchedulerSpec | None, SchedulerSpec | None]:
    """Resolve symmetric joint settings before any model is loaded."""

    gate_lr, graph_lr = _valid_lr(gate_lr), _valid_lr(graph_lr)
    if gate_lr is None and graph_lr is None:
        gate_lr = graph_lr = 1e-4
    elif gate_lr is None:
        gate_lr = graph_lr
    elif graph_lr is None:
        graph_lr = gate_lr
    if gate_scheduler is None and graph_scheduler is not None:
        gate_scheduler = graph_scheduler
    elif graph_scheduler is None and gate_scheduler is not None:
        graph_scheduler = gate_scheduler
    if not gate_frozen and gate_lr != graph_lr:
        raise ValueError("joint learning rates must be equal")
    if not gate_frozen and gate_scheduler != graph_scheduler:
        raise ValueError("joint schedulers must be equal")
    return gate_lr, graph_lr, gate_scheduler, graph_scheduler


def resolve_b_init(mode: str, *, has_gate_checkpoint: bool) -> str:
    if mode not in {"auto", "zero", "random"}:
        raise ValueError("b-init must be auto, zero, or random")
    if mode == "auto":
        return "zero" if has_gate_checkpoint else "random"
    return mode


def initialize_b_projection(
    scorer: GraphScorer, mode: str, *, has_gate_checkpoint: bool
) -> str:
    resolved = resolve_b_init(mode, has_gate_checkpoint=has_gate_checkpoint)
    if resolved == "zero":
        torch.nn.init.zeros_(scorer.b_proj.weight)
    else:
        scorer.b_proj.reset_parameters()
    return resolved


@dataclass(frozen=True)
class _PhaseResult:
    loss: float
    optimizer_steps: int
    delta_energy_share: float


def _parameters(modules) -> list[torch.nn.Parameter]:
    return [parameter for module in modules for parameter in module.parameters()]


def _graph_modules(scorer: GraphScorer):
    return scorer.a_proj, scorer.gin, scorer.b_proj, scorer.graph_builder


def build_adamw_optimizers(
    scorer: GraphScorer,
    *,
    gate_lr: float = 1e-4,
    graph_lr: float = 1e-3,
    weight_decay: float = 0.01,
    gate_frozen: bool = False,
    graph_frozen: bool = False,
):
    """Return ``(gate, graph)`` AdamW optimizers for disjoint parameter sets."""

    gate_lr, graph_lr = _valid_lr(gate_lr), _valid_lr(graph_lr)
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("weight decay must be finite and non-negative")
    gate_parameters = list(scorer.gates.parameters())
    graph_parameters = _parameters(_graph_modules(scorer))
    for parameter in gate_parameters:
        parameter.requires_grad_(not gate_frozen)
    for parameter in graph_parameters:
        parameter.requires_grad_(not graph_frozen)
    gate_optimizer = None
    graph_optimizer = None
    if not gate_frozen:
        gate_optimizer = torch.optim.AdamW(
            gate_parameters, lr=gate_lr, weight_decay=weight_decay
        )
    if not graph_frozen:
        graph_optimizer = torch.optim.AdamW(
            graph_parameters, lr=graph_lr, weight_decay=weight_decay
        )
    return gate_optimizer, graph_optimizer


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
    scorer: GraphScorer,
    config,
    model_id: str,
    prefix_ids: Tensor,
    prefill_chunk: int,
    data_cursor,
    wandb_run_id: str | None,
    gate_optimizer=None,
    graph_optimizer=None,
    gate_scheduler=None,
    graph_scheduler=None,
) -> Path:
    if kind not in {"best", "last"}:
        raise ValueError("checkpoint kind must be best or last")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    full_state = scorer.state_dict()
    payload = {
        "graph": _cpu_state(
            {
                name: value
                for name, value in full_state.items()
                if not name.startswith("gates.")
            }
        ),
        "gate": _cpu_state(scorer.gates.state_dict()),
        "graph_optimizer": (
            None if graph_optimizer is None else graph_optimizer.state_dict()
        ),
        "gate_optimizer": None if gate_optimizer is None else gate_optimizer.state_dict(),
        "graph_scheduler": (
            None if graph_scheduler is None else graph_scheduler.state_dict()
        ),
        "gate_scheduler": None if gate_scheduler is None else gate_scheduler.state_dict(),
        "config": copy.deepcopy(config),
        "model_id": model_id,
        "prefix_ids": _normal_cpu_tensor(prefix_ids),
        "prefill_chunk": prefill_chunk,
        "data_cursor": copy.deepcopy(data_cursor),
        "rng": _capture_rng_state(),
        "wandb_run_id": wandb_run_id,
    }
    path = output_dir / f"{kind}.pt"
    torch.save(payload, path)
    return path


def _restore_optional_state(component, state, name: str) -> None:
    if component is not None:
        if state is None:
            raise ValueError(f"checkpoint has no {name} state")
        component.load_state_dict(state)


def load_checkpoint(
    path,
    *,
    scorer: GraphScorer,
    gate_optimizer=None,
    graph_optimizer=None,
    gate_scheduler=None,
    graph_scheduler=None,
    restore_rng: bool = True,
):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = dict(payload["graph"])
    state.update({f"gates.{name}": value for name, value in payload["gate"].items()})
    scorer.load_state_dict(state, strict=True)
    _restore_optional_state(
        graph_optimizer, payload.get("graph_optimizer"), "graph optimizer"
    )
    _restore_optional_state(gate_optimizer, payload.get("gate_optimizer"), "gate optimizer")
    _restore_optional_state(
        graph_scheduler, payload.get("graph_scheduler"), "graph scheduler"
    )
    _restore_optional_state(
        gate_scheduler, payload.get("gate_scheduler"), "gate scheduler"
    )
    if restore_rng:
        _restore_rng_state(payload["rng"])
    return payload


def load_gate_checkpoint(scorer: GraphScorer, path) -> None:
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


class GraphTrainer:
    """Run memory-staged gate, graph, and joint optimization for one context."""

    def __init__(
        self,
        scorer: GraphScorer,
        *,
        gate_optimizer=None,
        graph_optimizer=None,
        gate_scheduler=None,
        graph_scheduler=None,
        token_microbatch_size: int = 1000,
        graph_microbatch_size: str | int | None = None,
    ) -> None:
        if (
            isinstance(token_microbatch_size, bool)
            or not isinstance(token_microbatch_size, int)
            or token_microbatch_size < 1
        ):
            raise ValueError("token microbatch size must be a positive integer")
        self.scorer = scorer
        self.gate_optimizer = gate_optimizer
        self.graph_optimizer = graph_optimizer
        self.gate_scheduler = gate_scheduler
        self.graph_scheduler = graph_scheduler
        self.token_microbatch_size = token_microbatch_size
        self.graph_microbatch_size = (
            scorer.graph_microbatch_size
            if graph_microbatch_size is None
            else graph_microbatch_size
        )
        resolve_graph_microbatch_size(
            self.graph_microbatch_size, scorer.num_layers, scorer.num_heads
        )

    @property
    def _device(self):
        return self.scorer.a_proj.weight.device

    @property
    def _dtype(self):
        return self.scorer.a_proj.weight.dtype

    @property
    def _loss_dtype(self):
        if self._dtype in {torch.float16, torch.bfloat16}:
            return torch.float32
        return self._dtype

    @property
    def _graph_modules(self):
        return _graph_modules(self.scorer)

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

    def _hidden(self, example, layer_ids, positions) -> Tensor:
        layer_ids_cpu = layer_ids.detach().to("cpu").tolist()
        hidden = torch.stack(
            [example.hidden_by_layer[layer_id][positions] for layer_id in layer_ids_cpu]
        )
        return hidden.to(device=self._device, dtype=self._dtype)

    def _initial_z(self, example, graph_ids, layer_ids) -> tuple[Tensor, Tensor]:
        z = torch.empty(
            graph_ids.numel(),
            example.sequence_length,
            self.scorer.a_proj.out_features,
            device=self._device,
            dtype=self._dtype,
        )
        hidden_energy = torch.zeros((), device=self._device, dtype=self._loss_dtype)
        with torch.no_grad():
            for positions in self._token_chunks(
                example.sequence_length, shuffle=False
            ):
                device_positions = positions.to(self._device)
                graph_hidden = self._hidden(example, layer_ids, positions)
                z[:, device_positions] = self.scorer.project_graph_nodes(
                    graph_hidden, graph_ids
                )
                hidden_energy += graph_hidden.to(self._loss_dtype).square().sum()
        return z, hidden_energy

    def _targets(self, example, layer_ids, head_ids, positions) -> Tensor:
        layer_ids_cpu = layer_ids.detach().to("cpu").tolist()
        head_ids_cpu = head_ids.detach().to("cpu").tolist()
        targets = torch.stack(
            [
                example.teacher_scores[layer_id, 0, head_id]
                for layer_id, head_id in zip(layer_ids_cpu, head_ids_cpu)
            ]
        )[:, positions]
        return targets.to(device=self._device)

    @staticmethod
    def _bce_sum(scores: Tensor, targets: Tensor) -> Tensor:
        loss_dtype = (
            torch.float32
            if scores.dtype in {torch.float16, torch.bfloat16}
            else scores.dtype
        )
        return F.binary_cross_entropy(
            scores.to(loss_dtype), targets.to(loss_dtype), reduction="sum"
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

    @staticmethod
    def _energy_share(delta_energy: Tensor, hidden_energy: Tensor) -> float:
        denominator = delta_energy + hidden_energy
        if denominator.item() == 0:
            return 0.0
        return (delta_energy / denominator).item()

    def train_gate_phase(self, example: TeacherExample) -> _PhaseResult:
        if self.gate_optimizer is None:
            raise ValueError("gate phase requires a gate optimizer")
        self._validate_example(example)
        for parameter in _parameters(self._graph_modules):
            parameter.grad = None

        cached = []
        hidden_energy = torch.zeros((), device=self._device, dtype=self._loss_dtype)
        with _frozen(_parameters(self._graph_modules)):
            with torch.no_grad():
                for graph_ids, layer_ids, head_ids in self.scorer.graph_batches(
                    microbatch_size=self.graph_microbatch_size
                ):
                    z, batch_hidden_energy = self._initial_z(
                        example, graph_ids, layer_ids
                    )
                    u = self.scorer.propagate_graph_nodes(z, graph_ids)
                    cached.append(
                        (
                            graph_ids.detach().to("cpu"),
                            layer_ids.detach().to("cpu"),
                            head_ids.detach().to("cpu"),
                            u.detach().to("cpu"),
                        )
                    )
                    hidden_energy += batch_hidden_energy

            total_loss = torch.zeros((), device=self._device, dtype=self._loss_dtype)
            delta_energy = torch.zeros_like(total_loss)
            steps = 0
            for positions in self._token_chunks(example.sequence_length, shuffle=True):
                self.gate_optimizer.zero_grad(set_to_none=True)
                denominator = self.scorer.num_graphs * positions.numel()
                for graph_ids_cpu, layer_ids_cpu, head_ids_cpu, u_cpu in cached:
                    graph_ids = graph_ids_cpu.to(self._device)
                    layer_ids = layer_ids_cpu.to(self._device)
                    head_ids = head_ids_cpu.to(self._device)
                    graph_hidden = self._hidden(example, layer_ids, positions)
                    u = u_cpu[:, positions].to(device=self._device, dtype=self._dtype)
                    scores, delta = self.scorer.score_mixed_graph_nodes(
                        graph_hidden, u, graph_ids, layer_ids, head_ids
                    )
                    numerator = self._bce_sum(
                        scores,
                        self._targets(example, layer_ids, head_ids, positions),
                    )
                    (numerator / denominator).backward()
                    total_loss += numerator.detach()
                    delta_energy += delta.detach().square().sum()
                self._step(self.gate_optimizer, self.gate_scheduler)
                steps += 1

        return _PhaseResult(
            loss=(total_loss / (self.scorer.num_graphs * example.sequence_length)).item(),
            optimizer_steps=steps,
            delta_energy_share=self._energy_share(delta_energy, hidden_energy),
        )

    def train_graph_phase(
        self, example: TeacherExample, *, joint: bool = False
    ) -> _PhaseResult:
        if self.graph_optimizer is None:
            raise ValueError("graph phase requires a graph optimizer")
        if joint and self.gate_optimizer is None:
            joint = False
        self._validate_example(example)
        self.graph_optimizer.zero_grad(set_to_none=True)
        if joint:
            self.gate_optimizer.zero_grad(set_to_none=True)
        else:
            for parameter in self.scorer.gates.parameters():
                parameter.grad = None

        total_loss = torch.zeros((), device=self._device, dtype=self._loss_dtype)
        delta_energy = torch.zeros_like(total_loss)
        hidden_energy = torch.zeros_like(total_loss)
        gate_context = _frozen(self.scorer.gates.parameters()) if not joint else _frozen(())
        with gate_context:
            for graph_ids, layer_ids, head_ids in self.scorer.graph_batches(
                microbatch_size=self.graph_microbatch_size
            ):
                z_value, batch_hidden_energy = self._initial_z(
                    example, graph_ids, layer_ids
                )
                z = z_value.detach().requires_grad_(True)
                u = self.scorer.propagate_graph_nodes(z, graph_ids)
                u_proxy = u.detach().requires_grad_(True)

                for positions in self._token_chunks(
                    example.sequence_length, shuffle=False
                ):
                    device_positions = positions.to(self._device)
                    graph_hidden = self._hidden(example, layer_ids, positions)
                    scores, delta = self.scorer.score_mixed_graph_nodes(
                        graph_hidden,
                        u_proxy[:, device_positions],
                        graph_ids,
                        layer_ids,
                        head_ids,
                    )
                    numerator = self._bce_sum(
                        scores,
                        self._targets(example, layer_ids, head_ids, positions),
                    )
                    (numerator / (self.scorer.num_graphs * example.sequence_length)).backward()
                    total_loss += numerator.detach()
                    delta_energy += delta.detach().square().sum()

                torch.autograd.backward(u, u_proxy.grad)
                z_gradient = z.grad.detach()
                for positions in self._token_chunks(
                    example.sequence_length, shuffle=False
                ):
                    device_positions = positions.to(self._device)
                    graph_hidden = self._hidden(example, layer_ids, positions)
                    z_recomputed = self.scorer.project_graph_nodes(
                        graph_hidden, graph_ids
                    )
                    torch.autograd.backward(
                        z_recomputed, z_gradient[:, device_positions]
                    )
                hidden_energy += batch_hidden_energy

        self._step(self.graph_optimizer, self.graph_scheduler)
        steps = 1
        if joint:
            self._step(self.gate_optimizer, self.gate_scheduler)
            steps += 1
        return _PhaseResult(
            loss=(total_loss / (self.scorer.num_graphs * example.sequence_length)).item(),
            optimizer_steps=steps,
            delta_energy_share=self._energy_share(delta_energy, hidden_energy),
        )

    def step_validation(self, loss: float) -> None:
        for scheduler in (self.gate_scheduler, self.graph_scheduler):
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(loss)

    def train_context(self, example: TeacherExample, *, mode: str = "two_phase"):
        if mode not in {"gate", "graph", "two_phase", "joint"}:
            raise ValueError("training mode must be gate, graph, two_phase, or joint")
        if mode == "gate" and self.gate_optimizer is None:
            raise ValueError("gate phase requires a gate optimizer")
        gate_result = None
        graph_result = None
        gate_steps = 0
        graph_steps = 0
        if mode in {"gate", "two_phase"} and self.gate_optimizer is not None:
            gate_result = self.train_gate_phase(example)
            gate_steps = gate_result.optimizer_steps
        if mode in {"graph", "two_phase"}:
            graph_result = self.train_graph_phase(example)
            graph_steps = 1
        elif mode == "joint":
            graph_result = self.train_graph_phase(example, joint=True)
            graph_steps = 1
            gate_steps = int(self.gate_optimizer is not None)
        return {
            "gate_loss": None if gate_result is None else gate_result.loss,
            "graph_loss": None if graph_result is None else graph_result.loss,
            "delta_energy_share": (
                graph_result.delta_energy_share
                if graph_result is not None
                else gate_result.delta_energy_share
            ),
            "gate_steps": gate_steps,
            "graph_steps": graph_steps,
        }
