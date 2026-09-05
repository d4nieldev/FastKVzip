"""Reusable selection and objective primitives for answer-supervised training."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real
from typing import TypeVar

import torch
import torch.nn.functional as F
from torch import Tensor, nn


T = TypeVar("T")


@dataclass(frozen=True)
class CompactedContext:
    """Physically compacted cache tensors and context-relative selections."""

    keys: tuple[Tensor, ...]
    values: tuple[Tensor, ...]
    indices: Tensor


@dataclass(frozen=True)
class AnswerObjective:
    """Differentiable answer loss plus token-weighted aggregation values."""

    loss: Tensor
    nll_sum: Tensor
    correct_tokens: int
    token_count: int

    @property
    def accuracy(self) -> float:
        return self.correct_tokens / self.token_count


@dataclass(frozen=True)
class ScoreGradientHealth:
    """L2 score-gradient norms over all, retained, and evicted positions."""

    overall: Tensor
    retained: Tensor
    evicted: Tensor


def _ratio_bounds(minimum: float, maximum: float) -> tuple[float, float]:
    if any(
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        for value in (minimum, maximum)
    ):
        raise ValueError("retention ratios must be finite numbers")
    minimum, maximum = float(minimum), float(maximum)
    if not 0 <= minimum <= maximum <= 1:
        raise ValueError("retention ratios must satisfy 0 <= minimum <= maximum <= 1")
    return minimum, maximum


def retention_ratio(
    kind: str,
    minimum: float,
    maximum: float,
    *,
    global_step: int,
    total_steps: int,
    rng: random.Random | None = None,
) -> float:
    """Choose one resumable uniform ratio or a clamped linear ratio."""

    minimum, maximum = _ratio_bounds(minimum, maximum)
    if (
        isinstance(global_step, bool)
        or not isinstance(global_step, int)
        or isinstance(total_steps, bool)
        or not isinstance(total_steps, int)
        or total_steps < 1
    ):
        raise ValueError("global step must be an integer and total steps must be positive")
    if kind == "uniform":
        if not isinstance(rng, random.Random):
            raise ValueError("uniform retention requires an explicit resumable RNG")
        return minimum + (maximum - minimum) * rng.random()
    if kind != "linear":
        raise ValueError("retention schedule must be uniform or linear")
    if total_steps == 1:
        return maximum
    step = min(max(global_step, 0), total_steps - 1)
    return maximum + (minimum - maximum) * step / (total_steps - 1)


def retained_token_count(token_count: int, ratio: float) -> int:
    """Return max(1, floor(ratio * N)) for a non-empty context."""

    if isinstance(token_count, bool) or not isinstance(token_count, int) or token_count < 1:
        raise ValueError("context token count must be positive")
    minimum, _ = _ratio_bounds(ratio, ratio)
    return max(1, math.floor(minimum * token_count))


def fallback_validation_split(requested: Sequence[T]) -> tuple[tuple[T, ...], tuple[T, ...]]:
    """Deduplicate in order and reserve the last ceil(10%) for validation."""

    unique = tuple(dict.fromkeys(requested))
    if len(unique) < 2:
        raise ValueError("fallback validation requires at least two requested examples")
    validation_count = math.ceil(len(unique) * 0.1)
    return unique[:-validation_count], unique[-validation_count:]


def _normal_tensor(tensor: Tensor) -> Tensor:
    if not tensor.is_inference():
        return tensor
    with torch.inference_mode(False):
        return tensor.clone()


def score_context_subgraphs(
    scorer: nn.Module,
    hidden_by_layer: Sequence[Tensor],
    *,
    subgraph_size: int | None = None,
    token_microbatch_size: int,
    graph_microbatch_size: int | None = None,
) -> Tensor:
    """Score independent context subgraphs and concatenate their raw scores."""

    hidden = tuple(
        _normal_tensor(value[0] if value.ndim == 3 and value.size(0) == 1 else value)
        for value in hidden_by_layer
    )
    if (
        not hidden
        or any(value.ndim != 2 for value in hidden)
        or len({value.size(0) for value in hidden}) != 1
    ):
        raise ValueError("hidden states must have shape [layers,tokens,hidden_dim]")
    token_count = hidden[0].size(0)
    if token_count < 1:
        raise ValueError("context must contain at least one token")
    if len(hidden) != int(scorer.num_layers):
        raise ValueError("hidden layer count does not match scorer")
    if (
        isinstance(token_microbatch_size, bool)
        or not isinstance(token_microbatch_size, int)
        or token_microbatch_size < 1
    ):
        raise ValueError("token microbatch size must be positive")
    if subgraph_size is None:
        subgraph_size = token_count
    if (
        isinstance(subgraph_size, bool)
        or not isinstance(subgraph_size, int)
        or subgraph_size < 1
    ):
        raise ValueError("subgraph size must be positive")

    chunks = []
    for start in range(0, token_count, subgraph_size):
        stop = min(start + subgraph_size, token_count)
        graph_hidden = torch.stack([value[start:stop] for value in hidden])
        scores = scorer(
            graph_hidden,
            microbatch_size=graph_microbatch_size,
            token_microbatch_size=min(token_microbatch_size, stop - start),
        )
        expected = (int(scorer.num_layers), 1, int(scorer.num_heads), stop - start)
        if tuple(scores.shape) != expected:
            raise ValueError(f"scorer returned {tuple(scores.shape)}, expected {expected}")
        chunks.append(scores)
    return torch.cat(chunks, dim=-1)


def global_topk_indices(raw_scores: Tensor, ratio: float) -> Tensor:
    """Select raw-score top-k independently per layer/head over the full context."""

    if raw_scores.ndim != 4 or raw_scores.size(-1) < 1:
        raise ValueError("raw scores must have shape [layers,batch,heads,context]")
    k = retained_token_count(raw_scores.size(-1), ratio)
    selected = torch.topk(raw_scores, k, dim=-1).indices
    return selected.sort(dim=-1).values


def _validate_cache(
    key_cache: Sequence[Tensor],
    value_cache: Sequence[Tensor],
    raw_scores: Tensor,
    context_range: tuple[int, int],
) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...], int, int]:
    keys, values = tuple(key_cache), tuple(value_cache)
    if len(keys) != raw_scores.size(0) or len(values) != len(keys):
        raise ValueError("cache layer count does not match raw scores")
    start, end = context_range
    if (
        isinstance(start, bool)
        or not isinstance(start, int)
        or isinstance(end, bool)
        or not isinstance(end, int)
        or start < 0
        or end <= start
    ):
        raise ValueError("context range must identify a non-empty cache slice")
    for key, value in zip(keys, values):
        if key.ndim != 4 or value.shape != key.shape:
            raise ValueError("cache tensors must match [batch,heads,tokens,dim]")
        if end > key.size(2):
            raise ValueError("context range exceeds the cache")
        if key.shape[:2] != raw_scores.shape[1:3]:
            raise ValueError("cache batch/head dimensions do not match raw scores")
    if end - start != raw_scores.size(-1):
        raise ValueError("raw score length must equal the context range")
    return keys, values, start, end


def compact_context_kv(
    key_cache: Sequence[Tensor],
    value_cache: Sequence[Tensor],
    raw_scores: Tensor,
    *,
    ratio: float,
    temperature: float,
    context_range: tuple[int, int],
    straight_through: bool = True,
) -> CompactedContext:
    """Globally select context K/V while preserving all cache tokens outside it."""

    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, Real)
        or not math.isfinite(temperature)
        or temperature <= 0
    ):
        raise ValueError("temperature must be finite and positive")
    keys, values, start, end = _validate_cache(
        key_cache, value_cache, raw_scores, context_range
    )
    indices = global_topk_indices(raw_scores, ratio)
    k = indices.size(-1)
    probabilities = (
        k * torch.softmax(raw_scores / float(temperature), dim=-1)
        if straight_through
        else None
    )
    compacted_keys, compacted_values = [], []
    for layer, (key, value) in enumerate(zip(keys, values)):
        index = indices[layer].unsqueeze(-1).expand(-1, -1, -1, key.size(-1))
        # Gather before concatenating: operations performed outside inference mode
        # produce normal tensors without cloning the full source cache.
        selected_key = torch.gather(key[:, :, start:end], 2, index)
        selected_value = torch.gather(value[:, :, start:end], 2, index)
        if probabilities is not None:
            selected_probability = torch.gather(probabilities[layer], 2, indices[layer])
            multiplier = torch.ones_like(selected_probability) + (
                selected_probability - selected_probability.detach()
            )
            selected_value = selected_value * multiplier.to(selected_value.dtype).unsqueeze(-1)
        compacted_keys.append(torch.cat((key[:, :, :start], selected_key, key[:, :, end:]), 2))
        compacted_values.append(
            torch.cat((value[:, :, :start], selected_value, value[:, :, end:]), 2)
        )
    return CompactedContext(tuple(compacted_keys), tuple(compacted_values), indices)


def answer_objective(logits: Tensor, token_ids: Tensor, *, answer_start: int) -> AnswerObjective:
    """Compute teacher-forced causal NLL and accuracy on answer targets only."""

    if logits.ndim != 3 or token_ids.ndim != 2 or logits.shape[:2] != token_ids.shape:
        raise ValueError("logits and token IDs must match [batch,sequence]")
    if (
        isinstance(answer_start, bool)
        or not isinstance(answer_start, int)
        or not 1 <= answer_start < token_ids.size(1)
    ):
        raise ValueError("answer start must identify at least one causal target token")
    answer_logits = logits[:, answer_start - 1 : -1]
    targets = token_ids[:, answer_start:]
    token_count = targets.numel()
    nll_sum = F.cross_entropy(
        answer_logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="sum"
    )
    correct = int((answer_logits.argmax(dim=-1) == targets).sum().item())
    return AnswerObjective(nll_sum / token_count, nll_sum, correct, token_count)


def score_gradient_health(score_gradient: Tensor, selected_indices: Tensor) -> ScoreGradientHealth:
    """Measure detached score gradients overall and by hard selection outcome."""

    if score_gradient.ndim != 4 or selected_indices.shape[:-1] != score_gradient.shape[:-1]:
        raise ValueError("score gradient and selected indices must share graph dimensions")
    gradient = score_gradient.detach().float()
    mask = torch.zeros_like(gradient, dtype=torch.bool)
    mask.scatter_(-1, selected_indices.to(mask.device), True)

    def norm(values: Tensor) -> Tensor:
        return values.norm() if values.numel() else gradient.new_zeros(())

    return ScoreGradientHealth(norm(gradient), norm(gradient[mask]), norm(gradient[~mask]))


def replay_score_gradients(
    scorer: nn.Module,
    hidden_by_layer: Sequence[Tensor],
    score_gradient: Tensor,
    selected_indices: Tensor,
    *,
    subgraph_size: int | None = None,
    token_microbatch_size: int,
    graph_microbatch_size: int | None = None,
) -> ScoreGradientHealth:
    """Replay an external answer-loss VJP through the differentiable graph scorer."""

    if score_gradient is None:
        raise ValueError("answer backward produced no score gradient")
    scores = score_context_subgraphs(
        scorer,
        hidden_by_layer,
        subgraph_size=subgraph_size,
        token_microbatch_size=token_microbatch_size,
        graph_microbatch_size=graph_microbatch_size,
    )
    if scores.shape != score_gradient.shape:
        raise ValueError("replayed scores do not match the external gradient")
    torch.autograd.backward(scores, score_gradient.detach().to(scores))
    return score_gradient_health(score_gradient, selected_indices)


def freeze_llm(model: nn.Module) -> nn.Module:
    """Freeze model weights without disabling input-gradient tracking."""

    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    return model


def validate_answer_training_model_identity(identity: str | object) -> str:
    """Accept Qwen/Llama identities and reject Gemma3 before model construction."""

    candidates = [identity] if isinstance(identity, str) else [type(identity).__name__]
    if not isinstance(identity, str):
        for owner in (identity, getattr(identity, "model", None)):
            if owner is None:
                continue
            candidates.extend(
                getattr(owner, name, None) for name in ("name", "name_or_path")
            )
            config = getattr(owner, "config", None)
            text_config = getattr(config, "text_config", config)
            candidates.extend(
                getattr(text_config, name, None)
                for name in ("model_type", "architectures")
            )
    normalized = " ".join(str(value).lower() for value in candidates if value is not None)
    if "gemma3" in normalized or "gemma-3" in normalized:
        raise ValueError("Gemma3 is not supported for answer-supervised graph training")
    if "qwen" in normalized:
        return "qwen"
    if "llama" in normalized:
        return "llama"
    raise ValueError("answer-supervised graph training supports only Qwen and Llama")
