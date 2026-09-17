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

## D4 — ⚪ Superseded by D8 — `kernel` was computed and retained on the GraNoLa branch

The original choice kept the folded Gram-and-projection product on every
branch, on the grounds that making it optional would spread empty-value checks
through several methods. Review found the cost understated and the argument
inconsistent with D3, which had already made the message features optional the
same way. Replaced by [D8](#d8--user-approved-amendment--the-folded-out-projection-is-skipped-when-granola-cannot-use-it).

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
  - 🟠 Plan gap. Implemented for the subgraph path only.
  - The whole-context replay branch still drops the caller's seed, so a GraNoLa
    run there backpropagates a different random draw than its forward. The user
    has stopped using that path and accepted it as-is rather than have it fixed.
  - No separate user approval for the rest.

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
  - The first version missed gate-only checkpoints written before this option
    existed: their GraNoLa width was copied from a graph width that does not
    exist, and every such checkpoint was then rejected. Fixed in [D9](#d9--user-approved-amendment--one-canonicalizer-and-one-set-of-valid-choices).
  - No separate user approval.

| Code reference | What this code does |
| --- | --- |
| [`uses_granola`](../../../prefill/graph/model.py) | Reports false when the scorer has no mixer, so the seed paths short-circuit. |
| [`load_checkpoint`](../../../prefill/graph/training.py) | Compares normalization config only when a mixer exists. |

## D8 — 🟢 User-approved amendment — the folded out projection is skipped when GraNoLa cannot use it

- **Background:**
  - The other branches reach hidden width in one product by folding the Gram
    matrix with the out projection ahead of time.
  - GraNoLa applies the out projection after its affine, which runs at graph
    width, so folding it in would reorder the model.
- **Decision:** The folded product is not built on the GraNoLa branch, and the
  prepared graph carries nothing in its place.
- **Plan gap or deviation:** Replaces [D4](#d4--superseded-by-d8--kernel-was-computed-and-retained-on-the-granola-branch), which kept it.
- **Reason and tradeoff:**
  - It is structurally inapplicable here, not merely unused, so building it
    invites a future reader to reach for it.
  - It costs about 60 MB per prepared graph at the production shape, copied to
    the host once per graph microbatch during the gate phase, plus one wasted
    product each time.
  - Cost: two empty-value checks, the same pattern D3 already uses.
- **Status:**
  - 🟢 User-approved amendment. Implemented after review.

| Code reference | What this code does |
| --- | --- |
| [`prepare_from_chunks`](../../../prefill/graph/model.py) | Builds the folded product only when the branch can use it. |
| [`test_granola_keeps_no_folded_out_projection`](../../../prefill/tests/test_graph_model.py) | Asserts GraNoLa carries none and BatchNorm still does. |

## D9 — 🟢 User-approved amendment — one canonicalizer and one set of valid choices

- **Background:**
  - Filling in the normalization settings a pre-normalization checkpoint lacks
    was written twice, once per training entry point, and the two copies
    disagreed about a checkpoint with no graph width.
  - The lists of valid choices were written out in five places, and the legacy
    activation marker in three.
- **Decision:** One canonicalizer and one set of choice constants live in the
  model module and are exported; both training entry points, the evaluator and
  the checkpoint loader use them.
- **Plan gap or deviation:** The plan did not mention this duplication, which
  predates the branch in part and was extended by it.
- **Reason and tradeoff:**
  - The duplication caused two real defects: gate-only checkpoints could not be
    loaded at all, and answer training could not resume a pre-normalization
    checkpoint at any seed but zero.
  - Adding a further normalization mode now touches one file instead of five.
  - Cost: the model module gains checkpoint-shaped logic that is not strictly
    about the model.
- **Status:**
  - 🟢 User-approved amendment. Implemented after review.

| Code reference | What this code does |
| --- | --- |
| [`canonical_normalization_config`](../../../prefill/graph/model.py) | The single fill-in, with the graph-width fallback that gate-only checkpoints need. |
| [`test_legacy_gate_only_checkpoint_loads_with_normalization_defaults`](../../../prefill/tests/test_graph_eval.py) | Loads a gate-only checkpoint written before this option existed. |
| [`test_legacy_stage_one_checkpoint_resumes_at_any_seed`](../../../prefill/tests/test_graph_answer_train_cli.py) | Resumes answer training from a pre-normalization checkpoint at a non-zero seed. |

## D10 — 🟢 User-approved amendment — mixer gradients refuse a sliced prepared graph

- **Background:**
  - The loop that pushes gradients back through the input projection walks
    chunk positions, and uses them both to index the gradient buffers and to
    fetch the matching context hidden states.
  - A prepared graph cut down to a token slice keeps its original token count
    while its projections shrink, and it does not record which tokens it kept.
- **Decision:** The loop walks the projections' own length, and refuses a
  prepared graph whose two lengths disagree.
- **Plan gap or deviation:** The plan did not discuss this helper, which the
  implementation extracted so both branches could share it. Sizing its loop from
  the stored full length was a regression that extraction introduced.
- **Reason and tradeoff:**
  - Review proposed only changing the length. That stops the crash but leaves
    the wrong pairing, because the slice's positions are not its context
    positions.
  - Refusing it is honest: nothing routes a slice here today, and supporting one
    would mean recording its positions, which nothing needs yet.
- **Status:**
  - 🟢 User-approved amendment. Implemented after review.

| Code reference | What this code does |
| --- | --- |
| [`_absorb_projection_gradients`](../../../prefill/graph/training.py) | Walks the projections' length and rejects a mismatch. |
| [`test_mixer_gradients_refuse_a_sliced_prepared_graph`](../../../prefill/tests/test_graph_training.py) | Passes a sliced graph and expects the refusal. |

## D11 — 🟠 Plan gap — the GraNoLa scale head starts from its own initialization

- **Background:**
  - BatchNorm starts with its scale at one, so the branch begins as a plain
    normalization.
  - The GraNoLa scale comes out of a small network with ordinary initialization,
    so it starts centred on zero with a random sign per feature.
- **Decision:** No identity-like start is added.
- **Plan gap or deviation:** The plan did not mention initialization.
- **Reason and tradeoff:**
  - The reference implementation uses the same shape with ordinary
    initialization and no offset, and the paper specifies only "learnable
    functions".
  - How each option gets off the ground is part of what a comparison between
    them measures, so giving one a hand-picked start would blur that.
  - Cost: GraNoLa's first steps are noisier than BatchNorm's.
- **Status:**
  - 🟠 Plan gap. Deliberate non-change, confirmed in review.

## D12 — 🟠 Plan gap — random node features are reproducible per device, not across devices

- **Background:**
  - The random node features are drawn from a generator attached to the tensor's
    own device.
  - Every reproducibility guarantee in this work rests on that draw.
- **Decision:** The draw stays device-local.
- **Plan gap or deviation:** The plan required reproducible draws but did not
  say across what.
- **Reason and tradeoff:**
  - Same seed, same device gives the same features, which is what resume and
    evaluation need.
  - The same seed on CPU and on GPU gives different features, so the float64
    CPU tests cannot detect a divergence there.
  - Drawing on the host and copying would fix it but adds a transfer of the full
    token-by-width draw on every prepare.
- **Status:**
  - 🟠 Plan gap. Known limitation, recorded rather than fixed.
