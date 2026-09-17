# Add a GPS mixer architecture alongside the current one

## Context

The whole-context mixer has exactly one architecture. Every token pair is
connected through an implicit low-rank similarity, aggregated in one step,
and folded back onto the hidden states as a small residual. It is a GIN-like
aggregation and nothing else can be tried.

The goal is a second architecture following GPS (arXiv 2205.12454), and a
setting that picks between the two, so the paper can compare them on the same
gate, the same data, and the same objectives.

## What the GPS architecture does

Per layer and KV head, exactly as today, each block runs:

    input  ->  local branch   (today's implicit low-rank aggregation)
           ->  global branch  (Performer attention)
           ->  sum, feedforward, back to the hidden states

In more detail:

1. Hidden states are projected into the low-rank graph space, and a sequence
   position encoding over the token's index inside its subgraph is added.
2. The **local branch** is the existing mechanism unchanged: a similarity
   kernel built from tied projections, aggregating a second projection.
3. The **global branch** is Performer attention: separate query, key and value
   projections, several heads, positive random features approximating a
   softmax, and the proper normalizing denominator. This is what makes it
   different from the branch beside it, which is an unnormalized linear
   attention with query and key tied.
4. Each branch gets its own residual connection and normalization.
5. The two are summed and passed through a feedforward block **at graph width,
   not hidden width**. At hidden width this would cost billions of parameters,
   because weights are independent per layer/head.
6. Steps 2-5 repeat for the configured depth, default one block.
7. The result is projected back to hidden width, passed through the same
   activation as today, scaled by the learned residual weight, and added to the
   hidden states. The gate therefore receives the same kind of correction it
   receives now, and nothing downstream of the mixer changes.

## How the choice is exposed

- A single architecture setting on both training entry points, defaulting to
  the current architecture.
- The choice, the depth, the attention head count, and the random-feature count
  are recorded in the checkpoint. Evaluation reads them from there, as it
  already reads every other mixer setting.
- The architecture is fixed once a checkpoint exists. Resuming or evaluating
  cannot switch it, matching how the other structural settings already behave.
- Checkpoints written before this change load as the current architecture.

## What GPS deliberately cannot do

- **It requires a subgraph size.** Both training entry points currently default
  to whole-context. With GPS and no subgraph size, they stop with a clear
  message rather than quietly falling back.
- **Evaluation matches training.** A GPS checkpoint scores in subgraphs, the
  size it was trained at. There is no whole-context GPS scoring path, so its
  normalization statistics never shift between training and evaluation.
- **It uses ordinary autograd** inside each subgraph. The hand-written streamed
  gradient path that makes whole-context training possible today is specific to
  the current formula; it is not touched, extended, or made generic.
- **Depth applies to GPS only.** Stacking the current architecture would mean
  deriving that streamed gradient math again for every extra block.

## What stays exactly as it is

- The current architecture: same numerics, same gradients, same checkpoints,
  same whole-context training, and still the default.
- The gate, the scoring, top-k selection, both objectives (teacher supervision
  and answer KL), the memory and speed knobs, resume, and the logged metrics.
- Random features are drawn once per run and saved with the checkpoint, so a
  reloaded model reproduces the scores it was trained to produce.

One thing to watch: the two architectures will not have equal parameter counts
at the same graph width. Parameter counts get reported at startup so you can
match them by hand when the comparison needs it.

## Files

- `prefill/graph/model.py` — the new block and the architecture selection.
- `prefill/graph/training.py` — route GPS to the autograd path, keep the
  existing streamed path for the current architecture.
- `prefill/graph/evaluation.py` — checkpoint schema, the new settings, expected
  parameter shapes per architecture.
- `prefill/train_graph.py`, `prefill/train_graph_answer.py` — the setting, its
  validation, and the refusals above.
- `prefill/README.md`, `docs/graph-fastkvzip-experiments.md` — document the
  setting and the subgraph requirement.

## Verification

- The current architecture is unchanged: scores and every parameter gradient
  match the pre-change implementation on a fixed seed.
- GPS subgraphs stay independent: changing the tokens of one subgraph does not
  move another subgraph's scores.
- GPS reduces to the current mechanism when its global branch is zeroed, which
  confirms the two branches are wired the way the diagram says.
- Performer attention matches exact softmax attention within tolerance on a
  small case, and every GPS parameter receives a gradient.
- Both refusals fire: GPS without a subgraph size, in both training entry
  points and in evaluation.
- Checkpoint round-trip: save, resume and evaluate reproduce scores; a
  pre-change checkpoint still loads and evaluates as the current architecture.
- Both training loops run end to end on a tiny synthetic model with GPS.
- The existing graph model, training, evaluation, answer-training and
  checkpoint suites all pass.
- Real training and evaluation need a CUDA GPU, so the automated tests use a
  tiny synthetic model on CPU. A cluster pilot is separate work.

## Delivery

Implement in a new worktree and branch from current `main`. Save this approved
plan and the implementation decisions under
`docs/changes/gps-mixer-architecture/`, update the two documents above, commit,
push, and open one PR.

---

# Addendum: approved plan for rebasing onto the merged granola main

*The plan above was approved and implemented first. The granola
normalization work then merged into main, and this second plan was
approved for bringing this branch onto it. Both are kept: the first
describes the feature, the second what the merge changed.*

## Context

The granola normalization work merged into `main` (PR #9). It makes the
implicit mixer's normalization a choice — none, batchnorm, or granola — and
reworks much of the code the GPS branch also touches.

The GPS branch (PR #32) adds a second mixer architecture. It was written
against the pre-granola main, so it no longer merges.

The goal is a branch that merges cleanly, keeps both features working, and is
ready to review again.

Two decisions already taken:

- **GPS refuses a normalization mode.** GPS normalizes inside its own blocks.
  Asking for batchnorm or granola together with GPS is an error, so a
  checkpoint never records a setting nothing applied.
- **Merge main into the branch**, one merge commit. No force-push, and the
  review history stays anchored.

## What the merge costs

Eighteen conflict hunks across seven files. The large artifacts — the GPS test
file, both change documents — do not conflict at all.

`model.py` 5, `evaluation.py` 5, `training.py` 3, `graph/__init__.py` 3,
`train_graph.py` 1, `train_graph_answer.py` 1, `README.md` 1.

## How the conflicts resolve

Most are unions of two additive changes. Three need judgement.

**Granola wins; my version is deleted.** Granola independently generalized the
prepared state and routed both slice sites through it — the same job my
`narrow_tokens` did. Its `select_tokens` also handles its new fields, which
mine does not. Delete `narrow_tokens` and my two call-site edits.

**Mine wins, absorbing granola's logic.** `build_adamw_optimizers` gets my
delegation to the mixer, and granola's normalization branching moves inside
`ImplicitGraphMixer.parameter_groups()`. `GraphTrainer._prepared_slice` keeps
my architecture-agnostic signature.

**Both kept side by side.** The mixer training phase: GPS takes the autograd
path, the implicit mixer keeps granola's streamed replay. Plus every import
block, key list, CLI flag, export, and documentation section.

## New work the merge exposes

Not conflict resolution. These break, or behave wrongly, once both sides sit
together.

**1. GPS crashes on granola's seed guard.** Every seed call site is guarded by
a scorer property that reads `mixer.normalization`. GPS has no such attribute,
so the guard itself raises before any check runs. Give `GPSGraphMixer` a
`normalization` of `"none"`: truthful, since GPS applies no mixer-level
normalization, and it makes the guard answer correctly. Its `prepare` and
`prepare_from_chunks` must also accept and ignore the `rnf_seed` and `offsets`
arguments the trainer now forwards.

**2. Refuse GPS with batchnorm or granola**, and record `"none"` for it. Put
the rule in the shared CLI helper both scripts already use for the GPS-only
settings, and add the flag to the list a gate-only run rejects.

**3. The architecture back-fill cannot live in granola's canonicaliser as it
stands.** That helper returns early when a config already names a
normalization, so a granola-era checkpoint would skip anything added to its
legacy branch. Compose instead: one entry point that fills the architecture
independently, then calls granola's helper. Delete my `recorded_architecture`
and the teacher script's own block, so all three consumers share one path.

This is the highest-risk area. Both PRs independently fixed "old checkpoints
must still load", and a careless merge reintroduces the bug the last review
caught.

**4. Settle the activation-order field.** Granola made it generic across
normalization modes and rewrites the legacy value during canonicalization. GPS
needs its own value in that scheme, checked against the architecture the
checkpoint claims.

**5. Check the gradient-norm helper.** Granola rewrote it to split a shared
group's energy across graphs, assuming a parameter's leading dimension divides
the graph count. Confirm every GPS parameter satisfies that.

## Files

- `prefill/graph/model.py` — constants, the composed canonicaliser, the mixer
  interface both architectures implement.
- `prefill/graph/training.py` — parameter grouping, the slice helper, the
  branch between streamed replay and autograd.
- `prefill/graph/evaluation.py` — checkpoint keys, expected shapes, the
  activation-order check.
- `prefill/train_graph.py`, `prefill/train_graph_answer.py` — flags, the
  refusal, the single canonicaliser call.
- `prefill/README.md`, `docs/graph-fastkvzip-experiments.md` — both feature
  sections, and that the two cannot be combined.
- `docs/changes/gps-mixer-architecture/decisions.md` — entries for the refusal,
  the shared canonicaliser, and what granola made redundant.

## Verification

- **Re-baseline the implicit-path check.** The proof that GPS leaves the
  implicit mixer bit-identical was taken against the old main. Re-take it
  against the new main, under each of the three normalization modes.
- **Both suites pass together** — main's graph tests and the GPS tests.
- **Old-checkpoint resume across the matrix**: gate-only, implicit, each
  normalization mode, and GPS. Neither PR's tests cover the other's axis, and
  a granola-era checkpoint is a case neither PR has seen.
- **GPS end to end after the merge** — a training step and an evaluation with
  granola's seed plumbing live, since that is the new failure mode.
- **The refusal fires** in both scripts, and a gate-only run rejects the flag.
- **Re-run the mutation sweep** with refreshed anchors, plus one for the
  refusal.

## Delivery

Merge `origin/main` into the branch and resolve in one merge commit. Commit the
new work separately, so the review can see what the merge changed and what it
did not. Push and update PR #32.
