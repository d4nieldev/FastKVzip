# Gate-space coupling and implicit self loops for the graph mixer

## Context

The stage-1 audit (docs/stage1-loss-identity-audit.md) found that the mixer's
hidden-width residual is filtered by the gate's projections and RMSNorms, so
the mixer barely influences the score, and that the implicit adjacency only
carries an uncontrolled self term (its diagonal). Dani proposed two changes:

1. **Gate-space coupling.** Instead of projecting the mixer output back to the
   LLM hidden size and adding it to the gate input, keep the message in the
   mixer space (width C) and add zero-initialized linear maps of it directly to
   the gate's normalized query and key vectors, and optionally to the gate's
   per-group logit bias.
2. **Explicit self loops** in the implicit adjacency: M = Y1 (Y1ᵀY2/T) + λ·Y2
   with a learnable per-graph λ.

Formula agreed on 2026-09-19 (per graph g = layer × KV head; d = gate dim,
G = query groups):

```
Y1 = X W1,  Y2 = X W2                          W1, W2 ∈ R^{D×C}
M  = Y1 (Y1ᵀ Y2 / T) + λ Y2                    λ learnable per graph (opt-in)
Z  = Normalize(M)                              none | batchnorm (context stats, width C) | granola
F  = LeakyReLU(γ ⊙ Z + β)                      γ, β ∈ R^C
q̃_{t,j} = q_{t,j} + U_qᵀ F_t                   U_q ∈ R^{C×d}, zero init, shared over groups j
k̃_t     = k_t     + U_kᵀ F_t                   U_k ∈ R^{C×d}, zero init
b̃_{t,j} = b_j     + u_jᵀ F_t                   u ∈ R^{C×G}, zero init, optional
ℓ_{t,j} = k̃_t·q̃_{t,j}/√d + b̃_{t,j};  sink logits use q̃;  p_t = mean_j 1/(1+Σ_s exp(ℓ^sink−ℓ))
```

## Decisions already made (2026-09-19)

| Question | Decision |
|---|---|
| Where does the injection live? | New orthogonal flag `--mixer-coupling {hidden, gate-space}`, default `hidden` (today's residual). Works with `--mixer-architecture implicit` and `gps`, and with every `--normalization` (none, batchnorm, granola). |
| Which injection variants? | `--injection-target {qk, logit, qk-logit}`, default `qk-logit`. Only meaningful under gate-space; refused otherwise. |
| Self loops | Implicit architecture only (GPS's block already has a residual around its aggregation; refused under gps). Opt-in via `--self-loop-init FLOAT`; without the flag no λ exists and behaviour is unchanged. Learnable per-graph λ initialized to the value. Available under both couplings. |
| Old path | Unchanged and bit-identical by default; old checkpoints load, train, resume and evaluate as before. |
| Scope | Stage 1 (`train_graph.py`), stage 2 (`train_graph_answer.py`) and evaluation (`eval_graph.py`) all support the new options in this PR. |
| Initialization | U_q, U_k, u start at zero so step 0 equals the gate-only score exactly. |
| Weight decay | U_q, U_k, u decay like projections; λ, γ, β do not. |
| Base | Branch from `origin/main` (32b98d0); the local checkout is 20 commits behind. |

## Design

All paths below are in `prefill/`. Line numbers refer to `origin/main`.

### Model (`graph/model.py`)

- **Constants and parsers.** `MIXER_COUPLINGS = ("hidden", "gate-space")`,
  `DEFAULT_MIXER_COUPLING = "hidden"`, `INJECTION_TARGETS = ("qk", "logit",
  "qk-logit")`, `DEFAULT_INJECTION_TARGET = "qk-logit"`, with parse helpers
  next to `parse_mixer_architecture` (:63). `canonical_checkpoint_config`
  (:131) also `setdefault("mixer_coupling", "hidden")` when `graph_dim` is set,
  so every old checkpoint reads as hidden coupling. `self_loop_init` and
  `injection_target` get no default: absent means not applied.
- **`GateInjection`** frozen dataclass: `query`, `key` (each `[graphs, tokens,
  d]` or None) and `logit` (`[graphs, tokens, G]` or None), with
  `select_tokens(index)` and `detached_to(device)` mirroring
  `PreparedImplicitGraph` (:459-509).
- **`GateSpaceInjection(nn.Module)`** owned by the mixer as `self.injection`:
  `PerGraphLinear` maps `query_proj`/`key_proj` (C → d) for targets containing
  `qk` and `logit_proj` (C → G) for targets containing `logit`, all
  zero-initialized after construction (PerGraphLinear's Kaiming init, :302-306,
  is overwritten). `forward(features [graphs,T,C], graph_ids) -> GateInjection`.
  Living under the mixer means `save_checkpoint`'s prefix split
  (`training.py:414-419`) stores it as mixer state and `parameter_groups`
  reaches it.
- **`ImplicitGraphMixer`** gains `coupling`, `injection_target`,
  `self_loop_init`, `gate_dim`, `query_groups` kwargs.
  - `self.self_loop = nn.Parameter(full((num_graphs,), self_loop_init))` or
    None. Shape `(num_graphs,)` satisfies `_mixer_gradient_energy`
    (`training.py:584-602`).
  - Under gate-space: `out_proj` and `alpha` are registered as None; batchnorm
    `gamma`/`beta` are `(groups, graph_dim)` instead of `(groups, hidden_dim)`;
    `self.injection = GateSpaceInjection(...)`. Under hidden: unchanged.
  - New helper `message(y1, y2, gram, graph_ids)` = `messages(y1, gram)` plus
    `λ·y2` when self loops exist. Used by `delta` (granola branch, :1087), by
    `_prepare_granola`'s statistics (:883), and by the gate-space path.
  - Hidden coupling with self loops: `raw = _raw(y1, kernel) + λ·(y2 @ Wᵀ)`
    (new helper `_raw_with_self_loop`), used in `delta` (:1089) and in the
    batchnorm statistics stream (:940-943).
  - `prepare_from_chunks` (:886): `keep_y2 = granola or self_loop is not None
    or coupling == "gate-space"`; `kernel` is None under gate-space; batchnorm
    statistics under gate-space are streamed over `message(...)` chunks (width
    C) instead of `_raw` (width D).
  - `activated` (:1033): under gate-space the granola branch returns
    `leaky_relu(transformed)` instead of `projected_activation`; batchnorm and
    none are already width-agnostic.
  - New `features(y1, y2, prepared, graph_ids)` = `activated(normalized(message))`
    at width C, and `correction_from_prepared(prepared)` returning the hidden
    delta (hidden coupling, today's `delta_from_prepared`) or
    `self.injection(features, ids)` (gate-space). `delta_from_prepared` stays
    for the hidden path.
  - `parameter_groups` (:1096): hidden adds `self_loop` to no-decay; gate-space
    decays `in_proj.weight` and the injection weights, no-decay gets
    `self_loop`, `gamma`, `beta` or the GraNoLa non-linear-weight set.
- **`GPSGraphMixer`** gains `coupling`, `injection_target`, `gate_dim`,
  `query_groups`. Under gate-space: no `out_proj`, no `alpha`,
  `self.injection`; `correction = injection(leaky_relu(x))` where `x` is the
  stack output that feeds `out_proj` today (:1592), in the reduction dtype.
  `PreparedGPSGraph.delta` becomes `correction: Tensor | GateInjection`;
  `select_tokens`/`detached_to` handle both. `parameter_groups` (:1568) skips
  the absent alpha.
- **Adapter** (`_HeadwiseGateAdapter`): `forward(gate, head, hidden, delta=None,
  injection=None)` and `forward_batch(gates, layer_ids, head_ids, hidden,
  correction=None)` dispatching on type. Injection lands after
  `normalize(queries, "q_norm")` / `normalize(keys, "k_norm")` (:1820-1821;
  oracle :1723/:1728): `queries = queries + injection.query.unsqueeze(2)`
  (broadcast over groups), `keys = keys + injection.key`, and
  `logits = logits + injection.logit` next to `gate.b` (:1824-1826; oracle
  :1731). Both operands are cast to `torch.promote_types(...)` so the
  fp32 master-dtype injection promotes the gate math exactly as the delta does
  today. Sink keys are untouched.
- **Scorer** (`ImplicitGraphScorer`): kwargs `mixer_coupling`,
  `injection_target`, `self_loop_init`; passes `gate_dim=self.gate_dim` and
  `query_groups=first_gate.ngroup` to the mixer. `score_prepared` (:2043)
  calls `correction_from_prepared` and returns `(scores, correction)`; every
  caller already discards the second element. `hidden_dtype` (:1988): under
  gate-space there is no delta to promote the gate input, so the hidden states
  are materialized in the master dtype, the gate-only rule, keeping the gate's
  projections at the same precision under both couplings.
  `scores_subgraphs_only` stays an `isinstance(GPSGraphMixer)` check.

### Training, stage 1 (`graph/training.py`)

- `_score_from_delta` (:882) becomes `_score_from_correction`;
  `_score_from_transformed` (:872) is coupling-aware: hidden → `alpha ·
  projected_activation` (unchanged), gate-space → `injection(leaky_relu(transformed))`.
- `train_mixer_phase` (:1271) keeps its dispatch: GPS → `_train_mixer_batch_autograd`
  (plain autograd through `score_prepared`, works for both couplings unchanged).
  Implicit → `_train_mixer_batch` (:998), which now routes:
  - gate-space (any normalization) and hidden+granola → the graph-width routine
    (today's `_train_granola_batch`, :1123, generalized to
    `_train_graph_width_batch`): y1/y2 proxies → gram recomputed from the
    proxies → `M = y1 @ gram + λ·y2` → the pre-activation `transformed`
    (batchnorm: `γ·ctxnorm(M)+β` with statistics recomputed live from M; none:
    `M`; granola: as today) → detach → per token chunk `_score_from_transformed`
    → backward into the injection weights and the chunk gradient → one backward
    from `transformed` into γ/β or the GraNoLa heads, λ and the proxies →
    `_absorb_projection_gradients(direct_y1, direct_y2, gram_gradient=None)`
    (:1210, unchanged).
  - hidden + batchnorm/none → today's two-pass streamed routine (:1021-1121).
    With self loops, `absorb_raw_gradient` (:1038) additionally backpropagates
    the chunk's `λ·(y2 @ Wᵀ)` term with autograd (a `y2` chunk proxy, the
    selected `out_proj` rows and `λ`), accumulating a `direct_y2_gradient`
    buffer that is passed to `_absorb_projection_gradients` instead of None.
    Without self loops the routine is byte-for-byte today's.
- Metrics: `train_graph.py:1138` logs `train/mean_alpha` only when the mixer
  has `alpha`, and `train/mean_self_loop` when it has `self_loop`. Same in
  `train_graph_answer.py:1434`.

### Stage 2 (`train_graph_answer.py`, `graph/answer_training.py`)

- Scoring and replay already go through `score_subgraph_batch` →
  `score_prepared` (`answer_training.py:195, :407`), so autograd reaches the
  injection weights with no change there.
- CLI mirrors stage 1: `--mixer-coupling`, `--injection-target`,
  `--self-loop-init` added to the parser and to `_MIXER_ONLY_FLAGS` (:77-99);
  the architecture block (:577-631) resolves them with `_pick` under the same
  strictness as `mixer_architecture`; `_make_components` (:1502) forwards them.

### CLI and checkpoint contract (`train_graph.py`, `graph/evaluation.py`)

- Flags in `train_graph.py` (:93-112) and `TrainingOptions` fields; resolution
  next to the architecture block (:381-465): `mixer_coupling` via `_pick`
  (default hidden), `injection_target` via `_pick` (default qk-logit) only under
  gate-space, `self_loop_init` via `_pick` with default None.
- Refusals, shared by both scripts like `reject_gps_only_options` (:1188):
  `--injection-target` requires `--mixer-coupling gate-space`; `--alpha-init`
  is refused under gate-space (no residual weight exists); `--self-loop-init`
  is refused under `--mixer-architecture gps`.
- `normalized_checkpoint_config` (:960): writes `mixer_coupling` whenever a
  mixer exists; `injection_target` only under gate-space; `self_loop_init`
  only when set; `alpha_init` only under hidden coupling ("record only what
  this run applies", :997-1006). `activation_order` is unchanged: the coupling
  is carried by its own key and the strict state-dict load.
- `load_checkpoint` (`training.py:498`): before the strict load, compare the
  saved coupling (default hidden), injection target and self-loop presence
  with the scorer and raise a named conflict, so a hidden checkpoint cannot be
  loaded into a gate-space scorer (or the reverse) with an opaque key error.
- `evaluation.py`: `alpha_init` leaves `_CONFIG_KEYS` (:35-58) and is required
  and numeric only under hidden coupling; `_validate_checkpoint` (:231) checks
  `mixer_coupling` ∈ couplings, `injection_target` present and valid under
  gate-space, `self_loop_init` numeric when present; `_expected_mixer_shapes`
  (:160) and `_expected_gps_shapes` (:117) become coupling-aware (no
  `out_proj`/`alpha` under gate-space, `gamma`/`beta` at width C, injection
  weights `mixer.injection.query_proj.weight (graphs, d, C)`,
  `key_proj.weight (graphs, d, C)`, `logit_proj.weight (graphs, G, C)` by
  target, and `mixer.self_loop (graphs,)` when set); `reconstruct_graph_scorer`
  (:444) forwards the three new settings and reads `alpha_init` with a default.

## Files

- `prefill/graph/model.py`: constants, `GateInjection`, `GateSpaceInjection`,
  both mixers, adapter, scorer.
- `prefill/graph/training.py`: scoring helpers, graph-width routine, self-loop
  term in the streamed routine, checkpoint conflict check.
- `prefill/graph/evaluation.py`: config keys, shape schema, validation,
  reconstruction.
- `prefill/train_graph.py`, `prefill/train_graph_answer.py`: flags, resolution,
  refusals, checkpoint config, metrics.
- `prefill/tests/test_gate_space_coupling.py` (new) plus small touches to
  `test_graph_model.py`, `test_graph_training.py`, `test_graph_eval.py`,
  `test_graph_train_cli.py`, `test_graph_answer_train_cli.py` where signatures
  or key inventories change.
- Docs: `prefill/README.md` (formula block and a coupling paragraph next to
  "Choosing the mixer architecture"), `docs/graph-fastkvzip-experiments.md`
  ("Change an experiment" table rows and refusal prose),
  `docs/experiment-results.md` (three default-table rows),
  `docs/graph-fastkvzip-decision-audit.md` (pointer blockquote for the
  self-connection and adapter rows), `docs/changes/gate-space-coupling/plan.md`
  and `decisions.md`.

## Tests (`prefill/tests/test_gate_space_coupling.py`, CPU, float64, tiny gates)

1. Dense reference: implicit + gate-space with a materialized `A + λI` for
   none, batchnorm and granola (graph and token adaptivity) reproduces the
   streamed scores.
2. Self-loop dense reference under hidden coupling: `(Y1 S + λ Y2) W` for all
   three normalizations.
3. Zero-init equality: a fresh gate-space scorer (implicit and gps, every
   target) scores exactly like the gate alone.
4. Target semantics: `logit` leaves queries and keys untouched; `qk` leaves the
   logit bias untouched.
5. Streamed training gradients equal full autograd: implicit gate-space ×
   {none, batchnorm, granola} × {qk-logit, logit}; hidden coupling with self
   loops × {none, batchnorm, granola}; joint mode included.
6. Token-microbatch and graph-microbatch invariance of scores and gradients
   under gate-space.
7. GPS + gate-space trains through the autograd path and still requires a
   subgraph size.
8. Checkpoint contract: recorded keys only when applied and `alpha_init`
   absent under gate-space; save/load round trip; evaluator validation and
   reconstruction reproduce scores; an old checkpoint without the coupling key
   loads as hidden; hidden↔gate-space and self-loop mismatches are refused with
   a named message.
9. CLI refusals in both scripts and resume cannot switch coupling or target.
10. Stage 2: `score_context_subgraphs` + `replay_score_gradients` under
    gate-space put gradients on the injection weights.
11. Decay membership covers every mixer parameter exactly once (gate-space,
    with and without self loops).
12. Metrics: `train/mean_self_loop` logged and `train/mean_alpha` absent under
    gate-space.
13. Existing suites stay green with the hidden default; the implicit hidden path
    is bit-identical (scores, losses, every parameter gradient) on a fixed seed.

## Verification

- `cd prefill && uv run --python 3.12 --with torch==2.7.0 --with pytest==9.1.1
  --with "numpy<2" python -m pytest tests/ -q` (the local CPU environment used
  for the audit); the ~22 tests that need optional dependencies fail the same
  way before the change.
- Mutation check on the new tests: remove the injection add, the self-loop
  term, or the zero init and confirm the matching test fails.
- Real training and evaluation need a CUDA GPU; a cluster pilot is separate
  work.

## Out of scope

- Any change to the loss (ranking-aware BCE, stage-2 relaxed mask).
- Token-specific cross-token features (windowed softmax, redundancy features).
- Self loops inside GPS's own implicit branch.
- Backfilling the missing GPS/normalization rows in `docs/experiment-results.md`
  beyond the three new rows.

## Delivery

Implement in a new worktree on `feature/gate-space-coupling` branched from
`origin/main`. Save this approved plan and the implementation decisions under
`docs/changes/gate-space-coupling/`, update the docs above, commit with the
repository's commit style, push, and open one PR against
`d4nieldev/FastKVzip`.
