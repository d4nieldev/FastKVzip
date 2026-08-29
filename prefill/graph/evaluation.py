"""Checkpoint reconstruction and streamed implicit-mixer inference."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from attention.gate import Weight

from .model import (
    ACTIVATION_ORDER,
    ImplicitGraphScorer,
    PreparedImplicitGraph,
    parse_compute_dtype,
    resolve_graph_microbatch_size,
)
from .training import load_checkpoint


_CONFIG_KEYS = (
    "model_id",
    "compute_dtype",
    "gate_dim",
    "gate_sink",
    "hidden_dim",
    "num_layers",
    "num_kv_heads",
    "query_groups",
    "graph_dim",
    "graph_microbatch_size",
    "token_microbatch_size",
    "gram_normalization",
    "normalization",
    "normalization_sharing",
    "granola_gnn_depth",
    "granola_mlp_depth",
    "granola_rnf_dim",
    "normalization_seed",
    "leaky_relu_slope",
    "activation_order",
    "alpha_init",
)

_NORMALIZATIONS = {"none", "batchnorm", "granola"}
_NORMALIZATION_SHARING = {"graph", "layer", "global"}
_LEGACY_ACTIVATION_ORDER = "batchnorm-leaky-relu"


@dataclass(frozen=True)
class EvaluationCheckpoint:
    payload: Mapping[str, object]
    config: Mapping[str, object]
    model_id: str
    prefix_ids: Tensor
    prefill_chunk: int
    compute_dtype: torch.dtype

    @property
    def token_microbatch_size(self) -> int:
        return int(self.config["token_microbatch_size"])

    @property
    def graph_microbatch_size(self) -> int:
        return int(self.config["graph_microbatch_size"])


def _positive_int(config: Mapping[str, object], name: str) -> int:
    value = config[name]
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"checkpoint {name} must be a positive integer")
    return value


def _state_tensor(state: Mapping[str, object], name: str) -> Tensor:
    value = state.get(name)
    if not isinstance(value, Tensor):
        raise ValueError(f"checkpoint state is missing tensor {name}")
    return value


def _canonical_checkpoint_config(config: Mapping[str, object]) -> dict[str, object]:
    """Upgrade pre-normalization metadata without weakening new validation."""

    canonical = copy.deepcopy(dict(config))
    if "normalization" not in canonical:
        canonical["normalization"] = "batchnorm"
        canonical["normalization_sharing"] = "graph"
        canonical["granola_gnn_depth"] = 1
        canonical["granola_mlp_depth"] = 1
        if "graph_dim" in canonical:
            canonical["granola_rnf_dim"] = canonical["graph_dim"]
        canonical["normalization_seed"] = 0
        if canonical.get("activation_order") == _LEGACY_ACTIVATION_ORDER:
            canonical["activation_order"] = ACTIVATION_ORDER
    return canonical


def _expected_mixer_shapes(
    config: Mapping[str, object], values: Mapping[str, int]
) -> dict[str, tuple[int, ...]]:
    """Return the exact mode- and sharing-specific mixer checkpoint schema."""

    layers, heads = values["num_layers"], values["num_kv_heads"]
    graphs = layers * heads
    hidden_dim, graph_dim = values["hidden_dim"], values["graph_dim"]
    sharing = config["normalization_sharing"]
    groups = graphs if sharing == "graph" else layers if sharing == "layer" else 1
    shapes = {
        "mixer.in_proj.weight": (graphs, 2 * graph_dim, hidden_dim),
        "mixer.out_proj.weight": (graphs, hidden_dim, graph_dim),
        "mixer.alpha": (graphs,),
    }
    normalization = config["normalization"]
    if normalization == "batchnorm":
        shapes.update(
            {
                "mixer.gamma": (groups, hidden_dim),
                "mixer.beta": (groups, hidden_dim),
            }
        )
        return shapes
    if normalization == "none":
        return shapes

    rnf_dim = values["granola_rnf_dim"]
    for block in range(values["granola_gnn_depth"]):
        prefix = f"mixer.granola_blocks.{block}"
        block_input = hidden_dim + rnf_dim if block == 0 else graph_dim
        shapes[f"{prefix}.linears.0.weight"] = (groups, graph_dim, block_input)
        for layer in range(1, values["granola_mlp_depth"]):
            shapes[f"{prefix}.norms.{layer - 1}.weight"] = (groups, graph_dim)
            shapes[f"{prefix}.norms.{layer - 1}.bias"] = (groups, graph_dim)
            shapes[f"{prefix}.linears.{layer}.weight"] = (
                groups,
                graph_dim,
                graph_dim,
            )
    for head in ("gamma", "beta"):
        prefix = f"mixer.granola_{head}_head"
        shapes.update(
            {
                f"{prefix}.linears.0.weight": (groups, graph_dim, graph_dim),
                f"{prefix}.linears.0.bias": (groups, graph_dim),
                f"{prefix}.norms.0.weight": (groups, graph_dim),
                f"{prefix}.norms.0.bias": (groups, graph_dim),
                f"{prefix}.linears.1.weight": (groups, hidden_dim, graph_dim),
                f"{prefix}.linears.1.bias": (groups, hidden_dim),
            }
        )
    return shapes


def _validate_checkpoint(payload: object) -> EvaluationCheckpoint:
    if not isinstance(payload, Mapping):
        raise ValueError("graph checkpoint must contain a mapping")
    saved_config = payload.get("config")
    if not isinstance(saved_config, Mapping):
        raise ValueError("graph checkpoint config must be a mapping")
    config = _canonical_checkpoint_config(saved_config)
    missing = [name for name in _CONFIG_KEYS if name not in config]
    if missing:
        raise ValueError(f"graph checkpoint config is missing: {', '.join(missing)}")
    model_id = config["model_id"]
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("checkpoint model_id must be a non-empty string")
    if payload.get("model_id") != model_id:
        raise ValueError("checkpoint model identifier disagrees with its config")
    compute_dtype = parse_compute_dtype(config["compute_dtype"])
    master_dtype = (
        torch.float32
        if compute_dtype in {torch.float16, torch.bfloat16}
        else compute_dtype
    )
    integer_names = (
        "gate_dim",
        "gate_sink",
        "hidden_dim",
        "num_layers",
        "num_kv_heads",
        "query_groups",
        "graph_dim",
        "graph_microbatch_size",
        "token_microbatch_size",
        "granola_gnn_depth",
        "granola_mlp_depth",
        "granola_rnf_dim",
    )
    values = {name: _positive_int(config, name) for name in integer_names}
    resolve_graph_microbatch_size(
        values["graph_microbatch_size"], values["num_layers"], values["num_kv_heads"]
    )
    if config["gram_normalization"] not in {"token-count", "none"}:
        raise ValueError("checkpoint gram_normalization is invalid")
    if config["normalization"] not in _NORMALIZATIONS:
        raise ValueError("checkpoint normalization is invalid")
    if config["normalization_sharing"] not in _NORMALIZATION_SHARING:
        raise ValueError("checkpoint normalization_sharing is invalid")
    normalization_seed = config["normalization_seed"]
    if (
        isinstance(normalization_seed, bool)
        or not isinstance(normalization_seed, int)
        or not 0 <= normalization_seed < 2**63
    ):
        raise ValueError(
            "checkpoint normalization_seed must be an integer from 0 to 2^63-1"
        )
    if config["activation_order"] != ACTIVATION_ORDER:
        raise ValueError("checkpoint activation order is invalid")
    for name in ("leaky_relu_slope", "alpha_init"):
        value = config[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"checkpoint {name} must be numeric")

    prefix_ids = payload.get("prefix_ids")
    if (
        not isinstance(prefix_ids, Tensor)
        or prefix_ids.ndim != 2
        or prefix_ids.size(0) != 1
        or prefix_ids.dtype != torch.long
    ):
        raise ValueError("checkpoint prefix_ids must have shape [1,prefix] and dtype long")
    prefill_chunk = payload.get("prefill_chunk")
    if isinstance(prefill_chunk, bool) or not isinstance(prefill_chunk, int) or prefill_chunk < 1:
        raise ValueError("checkpoint prefill_chunk must be a positive integer")

    mixer_state, gate_state = payload.get("mixer"), payload.get("gate")
    if not isinstance(mixer_state, Mapping) or not isinstance(gate_state, Mapping):
        raise ValueError("checkpoint mixer and gate states must be mappings")
    layers, heads = values["num_layers"], values["num_kv_heads"]
    hidden_dim = values["hidden_dim"]
    gate_dim, groups, sink = values["gate_dim"], values["query_groups"], values["gate_sink"]
    expected_mixer_shapes = _expected_mixer_shapes(config, values)
    for name, shape in expected_mixer_shapes.items():
        value = _state_tensor(mixer_state, name)
        if tuple(value.shape) != shape:
            raise ValueError(f"checkpoint {name} shape conflicts with normalized config")
        if value.dtype != master_dtype:
            raise ValueError("checkpoint mixer dtype is inconsistent")
    if set(mixer_state) != set(expected_mixer_shapes):
        raise ValueError("checkpoint mixer state has unexpected keys")

    for layer in range(layers):
        q_weight = _state_tensor(gate_state, f"{layer}.q_proj.weight")
        q_bias = _state_tensor(gate_state, f"{layer}.q_proj.bias")
        k_weight = _state_tensor(gate_state, f"{layer}.k_proj.weight")
        q_norm = _state_tensor(gate_state, f"{layer}.q_norm.weight")
        k_norm = _state_tensor(gate_state, f"{layer}.k_norm.weight")
        bias = _state_tensor(gate_state, f"{layer}.b")
        k_base = _state_tensor(gate_state, f"{layer}.k_base")
        if tuple(q_weight.shape) != (heads * groups * gate_dim, hidden_dim):
            raise ValueError("checkpoint q projection shape conflicts with normalized config")
        if tuple(q_bias.shape) != (heads * groups * gate_dim,):
            raise ValueError("checkpoint q projection bias conflicts with normalized config")
        if tuple(k_weight.shape) != (heads * gate_dim, hidden_dim):
            raise ValueError("checkpoint k projection shape conflicts with normalized config")
        if tuple(q_norm.shape) != (gate_dim,) or tuple(k_norm.shape) != (gate_dim,):
            raise ValueError("checkpoint gate norm shape conflicts with normalized config")
        if tuple(bias.shape) != (heads, 1, groups):
            raise ValueError("checkpoint gate bias shape conflicts with normalized config")
        if tuple(k_base.shape) != (heads, 1, sink, gate_dim):
            raise ValueError("checkpoint gate sink shape conflicts with normalized config")
        if any(
            tensor.dtype != master_dtype
            for tensor in (q_weight, q_bias, k_weight, q_norm, k_norm, bias, k_base)
        ):
            raise ValueError("checkpoint gate projection dtype is inconsistent")

    return EvaluationCheckpoint(
        payload=payload,
        config=copy.deepcopy(dict(config)),
        model_id=model_id,
        prefix_ids=prefix_ids.detach().to("cpu").clone(),
        prefill_chunk=prefill_chunk,
        compute_dtype=compute_dtype,
    )


def load_evaluation_checkpoint(
    path: str | Path, model_override: str | None = None
) -> EvaluationCheckpoint:
    """Load and validate graph metadata before constructing the LLM."""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint = _validate_checkpoint(payload)
    if model_override is not None and model_override != checkpoint.model_id:
        raise ValueError("--model must exactly match the graph checkpoint model_id")
    return checkpoint


def _model_dimensions(model) -> tuple[object, int, int, int, int]:
    config = getattr(model.config, "text_config", model.config)
    try:
        layers = int(config.num_hidden_layers)
        heads = int(config.num_key_value_heads)
        query_heads = int(config.num_attention_heads)
        hidden_dim = int(config.hidden_size)
    except AttributeError as error:
        raise ValueError("loaded model config lacks graph dimensions") from error
    if query_heads % heads:
        raise ValueError("loaded model attention heads are not divisible by KV heads")
    return config, layers, heads, query_heads // heads, hidden_dim


def reconstruct_graph_scorer(
    checkpoint: EvaluationCheckpoint, model
) -> ImplicitGraphScorer:
    """Reconstruct the implicit scorer from one current checkpoint."""

    config = checkpoint.config
    _, layers, heads, query_groups, hidden_dim = _model_dimensions(model)
    expected = (
        int(config["num_layers"]),
        int(config["num_kv_heads"]),
        int(config["query_groups"]),
        int(config["hidden_dim"]),
    )
    actual = (layers, heads, query_groups, hidden_dim)
    labels = ("layer count", "KV-head count", "query groups", "hidden size")
    for label, found, wanted in zip(labels, actual, expected):
        if found != wanted:
            raise ValueError(
                f"loaded model {label} {found} does not match checkpoint {wanted}"
            )
    if getattr(model, "gates", None) is not None:
        raise ValueError("loaded ModelKVzip must have no built-in gate")
    device = torch.device(model.device)
    gates = [
        Weight(
            layer,
            input_dim=hidden_dim,
            output_dim=int(config["gate_dim"]),
            nhead=heads,
            ngroup=query_groups,
            dtype=checkpoint.compute_dtype,
            sink=int(config["gate_sink"]),
        ).to(device)
        for layer in range(layers)
    ]
    scorer = ImplicitGraphScorer(
        gates,
        model.config,
        graph_dim=int(config["graph_dim"]),
        graph_microbatch_size=checkpoint.graph_microbatch_size,
        gram_normalization=str(config["gram_normalization"]),
        normalization=str(config["normalization"]),
        normalization_sharing=str(config["normalization_sharing"]),
        granola_gnn_depth=int(config["granola_gnn_depth"]),
        granola_mlp_depth=int(config["granola_mlp_depth"]),
        granola_rnf_dim=int(config["granola_rnf_dim"]),
        normalization_seed=int(config["normalization_seed"]),
        leaky_relu_slope=float(config["leaky_relu_slope"]),
        alpha_init=float(config["alpha_init"]),
        compute_dtype=checkpoint.compute_dtype,
    )
    load_checkpoint(checkpoint.payload, scorer=scorer, restore_rng=False)
    return scorer.eval()


def build_evaluation_runtime(checkpoint, *, model_factory=None):
    if model_factory is None:
        from model import ModelKVzip

        model_factory = ModelKVzip
    model = model_factory(checkpoint.model_id, kv_type="retain", gate_path_or_name="")
    if getattr(model, "gates", None) is not None:
        raise ValueError("ModelKVzip built-in gate must remain empty")
    return model, reconstruct_graph_scorer(checkpoint, model)


def _validate_hidden_cache(
    scorer: ImplicitGraphScorer,
    hidden_cache: Sequence[Tensor],
    start_idx: int,
    end_idx: int,
) -> None:
    if len(hidden_cache) != scorer.num_layers:
        raise ValueError(
            "unsupported hybrid/static hidden capture: "
            f"expected {scorer.num_layers} layers, got {len(hidden_cache)}"
        )
    if (
        isinstance(start_idx, bool)
        or not isinstance(start_idx, int)
        or isinstance(end_idx, bool)
        or not isinstance(end_idx, int)
        or start_idx < 0
        or end_idx <= start_idx
    ):
        raise ValueError("context cache indices must identify a non-empty range")
    for hidden in hidden_cache:
        if not isinstance(hidden, Tensor) or hidden.ndim != 3 or hidden.size(0) != 1:
            raise ValueError("cached hidden states must have shape [1,prefix+tokens,hidden_dim]")
        if hidden.device.type != "cpu":
            raise ValueError("cached hidden states must remain in CPU memory")
        if hidden.size(1) != end_idx:
            raise ValueError("cached hidden length must equal prefix plus the complete context")
        if hidden.size(2) != scorer.hidden_dim:
            raise ValueError(
                f"cached hidden dimension {hidden.size(2)} does not match "
                f"checkpoint {scorer.hidden_dim}"
            )


def _hidden_chunk(
    hidden_cache: Sequence[Tensor],
    layer_ids: Sequence[int],
    start: int,
    stop: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    return torch.stack(
        tuple(hidden_cache[layer_id][0, start:stop, :] for layer_id in layer_ids)
    ).to(device=device, dtype=dtype)


@torch.inference_mode()
def score_hidden_cache(
    scorer: ImplicitGraphScorer,
    hidden_cache: Sequence[Tensor],
    *,
    start_idx: int,
    end_idx: int,
    token_microbatch_size: int,
    graph_microbatch_size: int | None = None,
    rnf_seed: int | None = None,
) -> Tensor:
    """Score CPU hidden states without materializing a full mixer output."""

    if (
        isinstance(token_microbatch_size, bool)
        or not isinstance(token_microbatch_size, int)
        or token_microbatch_size < 1
    ):
        raise ValueError("token microbatch size must be a positive integer")
    _validate_hidden_cache(scorer, hidden_cache, start_idx, end_idx)
    token_count = end_idx - start_idx
    flat_score_batches = []
    for batch in scorer.graph_batches(microbatch_size=graph_microbatch_size):
        def chunks():
            for relative_start in range(0, token_count, token_microbatch_size):
                relative_stop = min(relative_start + token_microbatch_size, token_count)
                yield relative_start, _hidden_chunk(
                    hidden_cache,
                    batch.layer_ids,
                    start_idx + relative_start,
                    start_idx + relative_stop,
                    device=scorer.device,
                    dtype=scorer.compute_dtype,
                )

        prepared = scorer.mixer.prepare_from_chunks(
            chunks(),
            graph_ids=batch.graph_ids,
            token_count=token_count,
            token_microbatch_size=token_microbatch_size,
            rnf_seed=rnf_seed,
        )
        score_chunks = []
        for relative_start in range(0, token_count, token_microbatch_size):
            relative_stop = min(relative_start + token_microbatch_size, token_count)
            hidden = _hidden_chunk(
                hidden_cache,
                batch.layer_ids,
                start_idx + relative_start,
                start_idx + relative_stop,
                device=scorer.device,
                dtype=scorer.compute_dtype,
            )
            slice_prepared = prepared.select_tokens(
                torch.arange(
                    relative_start,
                    relative_stop,
                    device=prepared.y1.device,
                )
            )
            scores, _ = scorer.score_prepared(
                hidden,
                slice_prepared,
                layer_ids=batch.layer_ids,
                head_ids=batch.head_ids,
            )
            score_chunks.append(scores)
        flat_score_batches.append(torch.cat(score_chunks, dim=1))
    flat_scores = torch.cat(flat_score_batches, dim=0)
    return flat_scores.view(scorer.num_layers, scorer.num_heads, token_count).unsqueeze(1)


def protect_local_window(
    scores: Tensor,
    *,
    token_count: int,
    prefill_chunk: int,
    window_size: int,
) -> int:
    if window_size < 0:
        raise ValueError("window size must be non-negative")
    window = min(
        token_count,
        int(0.02 * token_count) if token_count < prefill_chunk else window_size,
    )
    if window > 0:
        scores[..., -window:] = scores.max()
    return window


def _clear_hidden_cache(kv) -> None:
    hidden_cache = getattr(kv, "hidden_cache", None)
    if hasattr(hidden_cache, "clear"):
        hidden_cache.clear()
    else:
        kv.hidden_cache = []


@torch.inference_mode()
def score_context_cache(
    kv,
    scorer: ImplicitGraphScorer,
    *,
    prefill_chunk: int,
    window_size: int,
    token_microbatch_size: int,
    graph_microbatch_size: int | None = None,
    rnf_seed: int | None = None,
) -> Tensor:
    """Assign scores to a retain cache and always release hidden states."""

    try:
        start_idx, end_idx = kv.start_idx, kv.end_idx
        token_count = end_idx - start_idx
        if getattr(kv, "ctx_len", token_count) != token_count:
            raise ValueError("cache context length conflicts with start/end indices")
        cache_device = torch.device(getattr(kv, "device", scorer.device))
        if cache_device != scorer.device:
            raise ValueError("graph scorer and KV cache must use the same device")
        scores = score_hidden_cache(
            scorer,
            kv.hidden_cache,
            start_idx=start_idx,
            end_idx=end_idx,
            token_microbatch_size=token_microbatch_size,
            graph_microbatch_size=graph_microbatch_size,
            rnf_seed=rnf_seed,
        )
        window = protect_local_window(
            scores,
            token_count=token_count,
            prefill_chunk=prefill_chunk,
            window_size=window_size,
        )
        kv.protected_window = window
        kv.score = scores
        print(f"Local window {window}")
        return scores
    finally:
        _clear_hidden_cache(kv)


def restore_checkpoint_prefix(model, prefix_ids: Tensor) -> None:
    """Undo the prefix mutation performed by each DataWrapper construction."""

    model.sys_prompt_ids = prefix_ids.to(model.device).clone()
