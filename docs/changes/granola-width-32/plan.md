# GraNoLa at width 32 with per-graph affine (PR #9)

## Context

PR #9 (`feature/granola-normalization`, branch commit `3e365f8`) adds a
`--normalization` flag so the mixer can use GraNoLa instead of BatchNorm. It has
been open since 2026-08-29 and `main` has moved 89 commits ahead, so it no longer
merges.

Reviewing it showed the implementation does not match the intended design:

- It predicts gamma and beta **per token**. We want **one pair per graph**.
- Those vectors are width 4096 (`hidden_dim`). We want width **32** (`graph_dim`).
- The GNN reads `A R W_o`, the finished width-4096 mixer output. We want it to read
  **R** (the width-32 message features) with the same implicit adjacency **A**.

Outcome: GraNoLa becomes a compact width-32 adaptive normalization, configurable
against BatchNorm, on a branch that merges cleanly into current `main`.

## Decisions already made

| Question | Decision |
|---|---|
| Where LeakyReLU sits | After `out_proj`, as in the paper. Only normalization moves earlier. |
| Norm statistics + affine granularity | Two coupled presets behind one flag. Default is across-token stats with per-graph affine. |
| BatchNorm branch | Unchanged. Stays at width 4096, after `out_proj`. The two branches deliberately differ. |
| Answer training and subgraph batching | Supported, not blocked. |

The GRANOLA paper (arXiv 2404.13344, Eq. 6 and 10) normalizes per node across
features, with per-node gamma/beta. Per-graph affine plus per-node normalization
would discard every token's magnitude with nothing to restore it, so the two
choices are coupled rather than independent.

## New flag

`--granola-adaptivity {graph,token}`, default `graph`. Config key
`granola_adaptivity`.

**`graph` (default).** Standardize `A R` over the token axis, per feature, per
graph — the same statistic the BatchNorm branch uses, at width 32. Mean-pool the
GNN output over tokens, then predict one gamma and one beta per graph, width 32.

**`token`.** LayerNorm over the 32 features per token, with per-token gamma and
beta of width 32. This is faithful GRANOLA, narrowed to 32.

Both modes produce identical parameter shapes; only the forward differs.

## The new GraNoLa forward

Per layer/head graph. `Y1 = X W_a`, `Y2 = R = X W_v`, both `[T, 32]`.
`gram = Y1ᵀ Y2 / c`, `A = Y1 Y1ᵀ / c`, `AR = Y1 @ gram`.

```
GNN input (block 0)  concat(Y2, rnf)        [T, 32 + r]
GNN block            P = first(in)
                     C = Y1ᵀ P / c          (A P computed as Y1 C, never forms T×T)
                     Q = P + Y1 C
                     H = finish(Q)          [T, 32]

graph mode           z = mean_t H[-1]       [1, 32]
token mode           z = H[-1]              [T, 32]

gamma, beta          heads(z)               width 32
normalized           graph: (AR - mu_t) * invstd_t,  stats [G, 32]
                     token: LayerNorm over the 32 features
delta                alpha * LeakyReLU( out_proj( gamma * normalized + beta ) )
```

`out_proj` moves inside the activation for this branch, so `delta` is still
`[G, T, 4096]` and the gate adapter contract is untouched.

## Implementation: plain autograd at width 32

The current branch hand-derives the whole GraNoLa backward because the GNN ran at
width 4096. At width 32 the activation footprint drops 128×, so the width-32
subgraph fits as an ordinary live autograd graph.

In `GraphTrainer._train_mixer_batch`, the GraNoLa branch becomes:

1. Detach `y1` and `y2` from the prepared state as autograd leaves.
2. Build `gram`, `AR`, the GNN, the affine and `normalized` live, at width 32.
3. Stream the width-4096 tail (`out_proj` → LeakyReLU → gate → BCE) per token
   chunk exactly as today, with `U = gamma*normalized+beta` **detached** and sliced.
   Collect each chunk's `U` gradient into one `[G, T, 32]` buffer.
4. One `torch.autograd.backward(U, u_gradient)` covers the GNN, both heads, the
   normalization, the pooling and `gram`.
5. Reuse the existing streamed `in_proj` VJP loop, fed from `y1.grad` and `y2.grad`.

This is exact: `U` is the only tensor crossing the chunk boundary, and no
parameter appears on both sides.

**Deleted, not extended** (roughly 200 lines):

- The GNN gradient replay in `training.py` — `replay_inputs`,
  `absorb_replay_inputs`, the two per-block passes, `contraction_proxy`,
  `previous_gradient`.
- `granola_hidden_gradient` and its `index_copy_`.
- `_granola_input_chunk` and the streamed two-pass `_prepare_granola` in
  `model.py`. Block-0 input no longer needs streaming at width 32.
- `_GranolaNormState.contractions` and the intermediate `hidden` layers. The
  replay was their only consumer.
- `absorb_raw_gradient` and the `_kernel(gram_proxy)` staging stay, but move under
  `if normalization != "granola"`. Leaving them live for GraNoLa would zero out
  `gram_gradient` and silently kill the `Y2` route.

This path should also be faster: today the GraNoLa branch evaluates the
width-4096 `Y1 @ kernel` product five to six times per mixer batch; the new one
evaluates a width-4096 product once forward and once backward.

**Memory.** Live set is about `22 * G * T * 32 * 4` bytes. At `G=112, T=4000`
that is ~1.3 GB. It only becomes a concern at `G=112` with contexts near 100k,
which the BatchNorm branch cannot run either. If that corner is needed later,
wrap each GNN block in `torch.utils.checkpoint`; not doing it now.

## Prepared state

`PreparedImplicitGraph` gains `y2` (`[G, T, 32]`, needed as the GNN input).
`_GranolaNormState` becomes:

| Field | Shape | Sliced by `select_tokens`? |
|---|---|---|
| `rnf` | `[G, T, r]` | yes |
| `token_hidden` | `[G, T, 32]`, `token` mode only | yes |
| `pooled` | `[G, 1, 32]`, `graph` mode only | **no** |
| `stats` | `[G, 32]` x2, `graph` mode only | **no** |

The pooled vector and the statistics must be computed once in `prepare` over the
**full** context. Pooling over a token chunk instead is the most likely silent
bug here, and it is what the microbatch-invariance tests exist to catch.

## Changes by file

**`prefill/graph/model.py`**
- `_PerGroupMLP` heads output `graph_dim` instead of `hidden_dim`; block-0 input
  becomes `graph_dim + granola_rnf_dim`.
- `_prepare_granola` collapses to a single width-32 pass plus the pooled or
  per-token readout.
- Split `normalized` / affine / `activated` so `activated` owns `out_proj` in the
  GraNoLa branch; `delta` reads `norm.pooled` or `norm.token_hidden`.
- `_sample_rnf` takes a per-row token start offset so stacked subgraphs of the
  same layer/head get distinct draws.
- `score_subgraph_batch` accepts and forwards `rnf_seed`.

**`prefill/graph/training.py`**
- The `_train_mixer_batch` rewrite above.
- `_score_from_normalized` becomes `_score_from_transformed`: it takes the
  width-32 affine output and applies `out_proj` for GraNoLa.
- Thread `rnf_seed` through the new `_optimizer_batches` / `_stacked_batch` /
  `_work_batches` paths, drawing once per example.

**`prefill/graph/answer_training.py`** (exists only on `main`)
- Thread `rnf_seed` through its own prepare/cache/slice path so stage-2 training
  works with GraNoLa.

**`prefill/graph/evaluation.py`**, **`prefill/train_graph.py`**
- Add `granola_adaptivity` to the config plumbing: argparse, `TrainingOptions`,
  both `_canonical_checkpoint_config` copies, `resolve_options`,
  `normalized_checkpoint_config`, `_CONFIG_KEYS`, `_validate_checkpoint`,
  `_expected_mixer_shapes`, `reconstruct_graph_scorer`, and
  `training.py:_checkpoint_normalization_config` plus `load_checkpoint`.
- Update `_expected_mixer_shapes`: block-0 input `graph_dim + rnf_dim`, head
  output `graph_dim`.

**Docs**
- Rewrite `docs/granola-normalization.md` (already on `main`, describes the old
  design and pins permalinks to `3e365f8`).
- Update the mixer section of `prefill/README.md`.

## Mergeability

Rebase `feature/granola-normalization` onto `main`, then force-push and update
PR #9.

`main` never touched `ImplicitGraphMixer`, so `model.py` auto-merges — and most of
it is being rewritten anyway. Real conflicts are 24 hunks in 8 files:

| File | Conflicts | Nature |
|---|---|---|
| `graph/training.py` | 6 | Structural. `main` added subgraph batching and optimizer-batch loops to the same functions GraNoLa rewrote. |
| `tests/test_graph_train_cli.py` | 4 | Both sides appended tests at the same spot. |
| `tests/test_graph_eval.py` | 5 | Mostly appended tests; two are genuine fixture edits. |
| `train_graph.py` | 3 | Two trivial import/insert collisions, one in `resolve_options` resume handling. |
| `graph/evaluation.py`, `eval_graph.py`, `graph/__init__.py`, `tests/test_graph_training.py` | 6 | Trivial. Both sides added a keyword argument or an `__all__` entry. |

One latent issue the textual merge hides: `main`'s `score_subgraph_batch` and
`_stacked_batch` replicate graph ids, and GraNoLa keys its RNF draw off graph id.
Without the token-offset change above, stacked subgraphs of one head would share
an identical RNF draw. Fixed as part of the subgraph-batching support.

## Testing

Rewrite, in `prefill/tests/`:
- `_dense_granola_delta` in `test_graph_model.py` — a fresh dense reference for
  both adaptivity modes.
- The granola shape tests in `test_graph_model.py` and `test_graph_eval.py`
  (`hidden+rnf` becomes `graph_dim+rnf`, head output becomes `graph_dim`).
- `test_granola_sharing_maps_the_whole_adaptive_module_by_scope`, whose probe
  writes a width-4096 bias.
- `test_granola_prepared_state_is_compact_and_singleton_safe`, for the new state.

Add:
- `test_streamed_gradients_match_full_autograd_for_new_normalizations`
  parametrized over both adaptivity modes, float64, `gnn_depth=2`, `mlp_depth=2`.
  This is the load-bearing correctness test.
- Token-microbatch invariance for `graph` mode specifically — it is the only new
  cross-token coupling and the only place a silent error can hide.
- `select_tokens` leaves `pooled` and `stats` untouched.
- GraNoLa through subgraph batching, asserting distinct RNF per stacked subgraph.
- GraNoLa through answer training.

Keep unchanged: the RNF seed reproducibility and eval-seed tests, the
`no T×T bmm` invariant test, every BatchNorm test, and the CLI choice tests.

## Verification

1. `python -m pytest prefill/tests/ -x` in the worktree.
2. Targeted: `pytest prefill/tests/test_graph_model.py prefill/tests/test_graph_training.py -k granola -v`.
3. A short real training run on CPU with a tiny model in both adaptivity modes,
   confirming the loss decreases and a checkpoint round-trips through
   `reconstruct_graph_scorer`.
4. `--normalization batchnorm` produces bit-identical results to `main` on the
   same seed, confirming the BatchNorm branch is untouched.

Cluster runs are not part of this PR.

## Out of scope

- The paper's `method.tex` still describes BatchNorm only. Not updating it here.
- No existing GraNoLa checkpoints exist, so no migration path is provided. The
  parameter shapes change and old GraNoLa checkpoints would be rejected.
- `torch.utils.checkpoint` for the GNN blocks, unless memory forces it.

## Records

Plan and decisions go to `docs/changes/granola-width-32/` in the same PR.
