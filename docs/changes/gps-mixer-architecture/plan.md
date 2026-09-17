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
