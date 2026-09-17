# GraNoLa width-32 decisions

Code decisions that filled a gap in, or deviated from, the approved
[plan](plan.md). Choices the plan already states are not repeated here.

| Marker and label | Meaning |
| --- | --- |
| 🔴 Plan deviation | Contradicts the approved plan, without explicit approval as an amendment. |
| 🟠 Plan gap | The plan left this code choice open. |
| 🟢 User-approved amendment | Approved separately after the plan. |
| ⚪ Superseded | Replaced; links to its replacement. |

## D1 — 🟠 Plan gap — shared normalization parameters get their gradient energy split across the graphs that use them

- **Background:**
  - The trainer logs one gradient norm per layer/head graph, by reshaping each
    mixer parameter so its first dimension is the graph count.
  - `--normalization-sharing layer` or `global` gives the normalization
    parameters fewer rows than there are graphs, so that reshape fails.
- **Decision:** A shared parameter's per-row energy is divided by the number of
  graphs in its group and repeated across them.
- **Plan gap or deviation:** The plan did not mention gradient-norm logging,
  which `main` added after this branch was written.
- **Reason and tradeoff:**
  - The gate already splits its shared RMSNorm weights evenly across heads, so
    this follows an existing convention rather than inventing one.
  - The alternative, skipping shared parameters, would silently under-report the
    mixer norm exactly when sharing is on.
- **Status:**
  - 🟠 Plan gap. Implemented.
  - No separate user approval.

| Code reference | What this code does |
| --- | --- |
| [`_mixer_gradient_energy`](../../../prefill/graph/training.py) | Splits a shared group's energy evenly across its graphs, and returns per-graph energy unchanged otherwise. |
| [`test_streamed_gradients_match_full_autograd_for_new_normalizations`](../../../prefill/tests/test_graph_training.py) | Runs the trainer with `normalization_sharing="global"`, which is the case that previously raised. |

## D2 — 🔴 Plan deviation — the branch's data-range config scheme is deleted rather than rebased

- **Background:**
  - The branch carried `DATA_RANGE_DEFAULTS`, which injected eight
    `train_data_*_idx` keys into resume validation.
  - `main` replaced that whole selection scheme with a context-count one and
    removed the keys and the `data_keys` function behind them.
- **Decision:** `DATA_RANGE_DEFAULTS`, its use in `_validate_resume_config`, and
  the test that called `train_graph.data_keys` are removed.
- **Plan gap or deviation:** The plan scoped the rebase to conflict resolution
  and did not anticipate deleting branch code that `main` had obsoleted.
- **Reason and tradeoff:**
  - Keeping it would inject keys into `saved` that the live config never has, so
    every resume would fail its equality check.
  - Cost: the branch loses an assertion about index-range defaults; the behavior
    it described no longer exists on `main`, so there is nothing to assert.
- **Status:**
  - 🔴 Plan deviation. Implemented.
  - Not separately approved; it removes dead code rather than changing agreed
    behavior.

| Code reference | What this code does |
| --- | --- |
| [`_validate_resume_config`](../../../prefill/train_graph.py) | Canonicalizes the saved config for normalization only, with no data-range injection. |

## D3 — 🟠 Plan gap — the retained message features are optional, not unconditional

- **Background:**
  - The GraNoLa GNN reads `Y2`, so the prepared graph has to keep it.
  - The gate phase copies every prepared graph to host memory, once per graph
    microbatch, so anything retained there is paid for on every branch.
- **Decision:** `PreparedImplicitGraph.y2` is `Tensor | None`, and only the
  GraNoLa branch fills it.
- **Plan gap or deviation:** The plan said the prepared graph "gains `y2`"
  without saying whether the other branches pay for it.
- **Reason and tradeoff:**
  - BatchNorm folds `Y2` into the Gram matrix and never needs it again, so
    retaining it would add `graphs x tokens x 32` floats to the default path for
    nothing.
  - Cost: two `None` checks in `select_tokens` and `detached_to`.
- **Status:**
  - 🟠 Plan gap. Implemented.
  - No separate user approval.

| Code reference | What this code does |
| --- | --- |
| [`prepare_from_chunks`](../../../prefill/graph/model.py) | Allocates the `y2` buffer only when the normalization is GraNoLa. |
| [`test_granola_prepared_state_is_compact_and_singleton_safe`](../../../prefill/tests/test_graph_model.py) | Asserts the GraNoLa branch retains `y2` at graph width. |

## D4 — 🟠 Plan gap — `kernel` is still computed and retained on the GraNoLa branch

- **Background:**
  - `kernel` fuses the Gram matrix with the out projection so the other branches
    reach hidden width in one product.
  - The GraNoLa branch normalizes before the out projection, so it never reads
    `kernel`.
- **Decision:** `kernel` is still built and carried for every branch.
- **Plan gap or deviation:** The plan listed what the GraNoLa path deletes but
  did not say whether `kernel` was among it.
- **Reason and tradeoff:**
  - Making it optional would push a `None` check into `delta`, `select_tokens`,
    `detached_to` and the staged backward for a tensor whose size does not grow
    with the context.
  - Cost: a GraNoLa run copies an unused `graphs x 32 x hidden` tensor to the
    host once per graph microbatch in the gate phase.
- **Status:**
  - 🟠 Plan gap. Implemented as a deliberate non-change.
  - No separate user approval.

## D5 — 🟠 Plan gap — a subgraph's random node features are keyed on its context offset, so stacked scoring no longer equals a whole-context call on the same slice

- **Background:**
  - Subgraph batching packs several context blocks of one layer/head into one
    call, repeating the graph id for each block.
  - The random node features were keyed on the graph id alone, so every block of
    a layer/head received an identical draw.
- **Decision:** The draw is keyed on the graph id and the block's start offset.
- **Plan gap or deviation:**
  - The plan asked for "distinct RNF per stacked subgraph" without saying what
    it should stay equal to.
  - Under this key a packed subgraph and a fresh whole-context call on the same
    token slice now differ, because the latter has no offset.
- **Reason and tradeoff:**
  - Each block is a separate graph, so independent draws match how the GRANOLA
    paper samples per graph, and identical draws would make two blocks with the
    same content indistinguishable to the normalization GNN.
  - The property that actually matters is preserved: how blocks are packed into
    calls does not change the scores.
  - Cost: for GraNoLa only, a subgraph's scores depend on where it sits in the
    context, which is a real behavior change from the branch's earlier state.
- **Status:**
  - 🟠 Plan gap. Implemented.
  - No separate user approval.

| Code reference | What this code does |
| --- | --- |
| [`_sample_rnf`](../../../prefill/graph/model.py) | Mixes the row's offset into the per-graph seed when offsets are supplied. |
| [`test_granola_gives_each_stacked_subgraph_its_own_random_features`](../../../prefill/tests/test_graph_eval.py) | Scores a context whose two halves are identical and asserts GraNoLa tells them apart. |
| [`test_granola_subgraph_scoring_packs_without_changing_scores`](../../../prefill/tests/test_graph_eval.py) | Asserts the packing of subgraphs into calls does not move the scores. |

## D6 — 🟠 Plan gap — answer training gets the full normalization config and a caller-chosen seed, not a threaded prepare path

- **Background:**
  - Answer training scores a context, then replays that same forward after the
    language-model backward to avoid retaining its activations.
  - It does not call the mixer's prepare path directly; it calls
    `score_subgraph_batch`.
- **Decision:**
  - `AnswerTrainingOptions` gains all seven normalization fields, with the same
    CLI flags and resume rules as stage-1 training.
  - The training step draws one RNF seed per context and hands the same seed to
    the forward and to the replay.
- **Plan gap or deviation:** The plan described threading the seed "through its
  own prepare/cache/slice path", which that module does not have.
- **Reason and tradeoff:**
  - Without a shared seed the replay would backpropagate a different set of
    random node features than the forward it is correcting, which produces wrong
    gradients rather than an error.
  - Without the config fields the checkpoint writer raises, because it reads
    them off the options object for both trainers.
  - Cost: the seed becomes part of the answer-training step's signature.
- **Status:**
  - 🟠 Plan gap. Implemented.
  - No separate user approval.

| Code reference | What this code does |
| --- | --- |
| [`resolve_rnf_seed`](../../../prefill/graph/model.py) | Settles one seed for a whole context, drawing a fresh one only in training mode. |
| [`replay_score_gradients`](../../../prefill/graph/answer_training.py) | Accepts the forward's seed instead of drawing its own. |
| [`test_granola_answer_replay_matches_a_direct_backward`](../../../prefill/tests/test_graph_eval.py) | Compares replayed mixer gradients against a direct backward through the same forward. |

## D7 — 🟠 Plan gap — a scorer without a mixer skips the normalization checks entirely

- **Background:**
  - `main` added a gate-only mode in which the scorer has no mixer at all and
    `graph_dim` is `None`.
  - The branch reads `scorer.mixer.normalization` in several places that a
    gate-only scorer also reaches.
- **Decision:** A `uses_granola` property gates those reads, the checkpoint
  normalization comparison is skipped when there is no mixer, and the expected
  mixer shapes stay empty in that case.
- **Plan gap or deviation:** The plan did not mention gate-only scorers, which
  landed on `main` after this branch was written.
- **Reason and tradeoff:**
  - A gate-only checkpoint has no mixer state, so there is no normalization for
    it to agree on.
  - The alternative, giving a mixer-less scorer a default normalization, would
    invent a setting that nothing applies.
- **Status:**
  - 🟠 Plan gap. Implemented.
  - No separate user approval.

| Code reference | What this code does |
| --- | --- |
| [`uses_granola`](../../../prefill/graph/model.py) | Reports false when the scorer has no mixer, so the seed paths short-circuit. |
| [`load_checkpoint`](../../../prefill/graph/training.py) | Compares normalization config only when a mixer exists. |
