# Implementation decisions — gate-space coupling and implicit self loops

Choices made while implementing [the approved plan](plan.md) that the plan left
open or that departed from it. Choices the plan already states are not repeated
here.

| Marker and label | Meaning |
| --- | --- |
| 🔴 Plan deviation | Contradicts the approved plan without separate approval. |
| 🟠 Plan gap | The plan left this code choice open. |
| 🟢 User-approved amendment | Approved separately, after the plan. |
| ⚪ Superseded | Replaced by a later decision. |

## D1 — 🔴 Plan deviation — stage 2 keeps its fixed metric key set and reports absent weights as 0.0

- **Background:**
  - Stage 2 asserts that every training log call emits exactly one fixed set of keys, and already logs `train/mean_alpha` as `0.0` for a gate-only run that has no alpha.
  - The plan asked both scripts to log `train/mean_alpha` only when the residual weight exists and `train/mean_self_loop` only when a self loop exists.
- **Decision:** Stage 2 always logs both keys; a weight the run does not have reports `0.0`. Stage 1, which has no fixed key set, logs each key only when the weight exists, as planned.
- **Plan gap or deviation:** The plan's "Same in `train_graph_answer.py`" contradicts stage 2's exact-allowlist contract, which the plan did not mention.
- **Reason and tradeoff:**
  - Keeping the allowlist exact preserves the existing guarantee that every stage-2 run writes the same W&B columns, which the resume and comparison tooling relies on.
  - The cost is that a `0.0` in `train/mean_alpha` under gate-space means "no such weight", the same convention the gate-only runs already use; a reader has to know the coupling to interpret it.
- **Status:**
  - 🔴 Plan deviation. Implemented.
  - No separate user approval; flagged in the PR.

| Code reference | What this code does |
| --- | --- |
| [`train_graph_answer.py` metric dict](../../../prefill/train_graph_answer.py) | Emits both keys, with `0.0` when the mixer lacks the weight. |
| [`TRAIN_LOG_KEYS`](../../../prefill/train_graph_answer.py) | Adds `train/mean_self_loop` to the exact allowlist. |
| [`test_stage_one_logs_the_self_loop_mean_instead_of_alpha_under_gate_space`](../../../prefill/tests/test_gate_space_coupling.py) | Fails if stage 1 stops logging conditionally. |

## D2 — 🟠 Plan gap — the GPS prepared state keeps a read-only `delta` accessor

- **Background:**
  - The GPS mixer's prepared state used to hold one field, `delta`, the hidden-width correction.
  - The plan renames it to `correction`, which can now also be a gate injection.
- **Decision:** The field is `correction`, and a `delta` property returns it when it is a hidden-width tensor and raises a named error when it is an injection.
- **Plan gap or deviation:** The plan did not say whether callers reading `.delta` should keep working.
- **Reason and tradeoff:**
  - Three existing GPS tests and any external caller read `.delta`; the accessor keeps them valid without pretending an injection is a delta.
  - The cost is one more name for the same value on the hidden path.
- **Status:**
  - 🟠 Plan gap. Implemented.
  - No separate user approval.

| Code reference | What this code does |
| --- | --- |
| [`PreparedGPSGraph.delta`](../../../prefill/graph/model.py) | Returns the tensor correction or raises for an injection. |
| [`test_the_gps_prepared_state_detaches_and_moves_like_the_implicit_one`](../../../prefill/tests/test_gps_mixer.py) | Unchanged; still reads `.delta`. |

## D3 — 🟠 Plan gap — the evaluator tolerates a stray `alpha_init` in a gate-space config

- **Background:**
  - The checkpoint writer records `alpha_init` only under the hidden coupling, as planned.
  - The evaluator validates a config on load and could also reject a key that names nothing.
- **Decision:** On read, `alpha_init` is required and checked only under the hidden coupling; if present under gate-space it is ignored rather than refused.
- **Plan gap or deviation:** The plan fixed what the writer emits, not what the reader rejects.
- **Reason and tradeoff:**
  - A hand-edited or externally produced config with a harmless extra key still evaluates, matching how the evaluator treats other unused keys.
  - The cost is that a stray value is not flagged; nothing reads it, so it cannot change a score.
- **Status:**
  - 🟠 Plan gap. Implemented.
  - No separate user approval.

| Code reference | What this code does |
| --- | --- |
| [`_validate_checkpoint`](../../../prefill/graph/evaluation.py) | Requires and type-checks `alpha_init` only when the coupling is hidden. |
| [`test_coupling_and_self_loop_mismatches_are_refused_by_name`](../../../prefill/tests/test_gate_space_coupling.py) | Loads a gate-space config carrying `alpha_init` and asserts it is accepted. |

## D4 — 🔴 Plan deviation — no independent dense reference for GraNoLa

- **Background:**
  - The plan's first two tests asked for materialized-adjacency references of the scores for none, batchnorm and granola, under gate-space and under the hidden coupling with self loops.
  - Reproducing GraNoLa's GNN, readout and affine heads densely in the new module would duplicate the dense helper the existing model tests already keep.
- **Decision:** Both dense references cover `none` and `batchnorm`; GraNoLa is covered by the streamed-versus-autograd gradient tests (both couplings, with self loops), the microbatch-invariance test, and the checkpoint round trip, on top of the existing dense GraNoLa tests of the shared message path.
- **Plan gap or deviation:** The plan listed granola in both dense-reference tests.
- **Reason and tradeoff:**
  - The gate-space change to GraNoLa is the tail after its affine (no out projection, then the injection); that tail is exactly what the dense none/batchnorm references and the gradient test exercise.
  - The cost is that a defect confined to how GraNoLa's affine output is fed into the tail would be caught by the gradient and equivalence tests but not by a formula-level check.
- **Status:**
  - 🔴 Plan deviation. Implemented as described.
  - No separate user approval; flagged in the PR.

| Code reference | What this code does |
| --- | --- |
| [`test_gate_space_scores_match_the_dense_self_loop_formula`](../../../prefill/tests/test_gate_space_coupling.py) | Dense reference for none and batchnorm. |
| [`test_gate_space_streamed_training_matches_full_autograd`](../../../prefill/tests/test_gate_space_coupling.py) | GraNoLa coverage through gradients against plain autograd. |
