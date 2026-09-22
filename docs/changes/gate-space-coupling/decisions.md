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

## D5 — 🟠 Plan gap — both ways into the gate adapter now materialize the hidden states the same way

- **Background:**
  - The scorer keeps every gate parameter at the master dtype and declares, through `hidden_dtype`, the dtype the gate's input should arrive in.
  - `score_prepared` casts to it; the trainer reaches the same adapter directly and passed the context hidden states on at the compute dtype instead.
  - Under the hidden coupling the delta is accumulated at the master dtype, so adding it promoted the gate input either way and the two call sites agreed by accident.
- **Decision:** The trainer's call site casts to the scorer's `hidden_dtype`, and the adapter's per-layer normalization widens its output tensor instead of failing when a norm returns a wider dtype than its input.
- **Plan gap or deviation:** The plan fixed what `hidden_dtype` should return under the new coupling but not which call sites have to honour it.
- **Reason and tradeoff:**
  - The gate-space coupling adds nothing to the gate input, so nothing promotes it any more and the disagreement became a real one: a bfloat16 gate raised inside the gate's normalization on the first training step.
  - Casting at the call site keeps the gate's projections at one precision whichever way they are reached; widening inside the adapter keeps the batched path and the single-head oracle in agreement for any dtype pairing, rather than one raising where the other succeeds.
  - The cost is one cast on a path that did not have one, which is a no-op whenever the two dtypes already agree.
- **Status:**
  - 🟠 Plan gap. Implemented after a GPU pilot (job 21476066) failed on it.
  - No separate user approval; the user approved fixing it before resubmitting.

| Code reference | What this code does |
| --- | --- |
| [`_score_from_correction`](../../../prefill/graph/training.py) | Materializes the hidden states in the scorer's gate-input dtype, as `score_prepared` does. |
| [`_HeadwiseGateAdapter.forward_batch`](../../../prefill/graph/model.py) | Widens the normalized tensor rather than raising when a norm returns a wider dtype. |
| [`test_both_ways_into_the_gate_score_a_low_precision_run_identically`](../../../prefill/tests/test_gate_space_coupling.py) | Fails if the two call sites compute a low-precision run's scores at different precisions. |
| [`test_the_adapter_agrees_with_its_oracle_when_the_norm_widens_the_dtype`](../../../prefill/tests/test_gate_space_coupling.py) | Fails if the batched path raises where the single-head oracle succeeds. |

## D6 — 🟢 User-approved amendment — the injection maps can start at a nonzero scale

- **Background:**
  - The plan fixed the injection maps at zero so a run begins at exactly the released gate's scores.
  - The mixer's own gradient arrives through those maps, so at zero the mixer body receives exactly nothing and cannot train until the maps grow.
  - The first GPU pilot showed it: the maps moved 1.8e-3 over an epoch while the self-loop weight and the normalization scale moved around 3e-6.
- **Decision:** `--injection-init` sets the standard deviation the maps start at, defaulting to 0.0, which is the planned behaviour.
- **Plan gap or deviation:** The plan specified zero initialization and said nothing about making the scale a choice.
- **Reason and tradeoff:**
  - A nonzero start wakes the whole mixer immediately, which a fixed-length run needs; measured, the body's gradient goes from exactly 0.0 to nonzero.
  - The cost is the exact gate-only start: a nonzero scale perturbs a well-trained gate, and the repo's own alpha sweep shows a random perturbation starts worse and is partly undone.
  - A scalar gain in front of a randomly initialized map was rejected: at gain zero the map's gradient is exactly zero, the same defect as the `alpha ≡ 0` checkpoints, so the scale belongs on the maps themselves.
- **Status:**
  - 🟢 User-approved amendment. Dani asked for an alpha-like scale after seeing the pilots' near-identical loss curves.
  - Default unchanged, so nothing already planned or recorded behaves differently.

| Code reference | What this code does |
| --- | --- |
| [`GateSpaceInjection`](../../../prefill/graph/model.py) | Starts the maps at the requested deviation, or at zero when it is 0. |
| [`load_checkpoint`](../../../prefill/graph/training.py) | Reads the scale a checkpoint declares, treating its absence as zero so earlier gate-space checkpoints still load. |
| [`test_zero_initialized_maps_leave_the_mixer_body_without_gradient`](../../../prefill/tests/test_gate_space_coupling.py) | Pins the reason the flag exists: at zero the body's gradient is exactly 0.0. |
| [`test_a_nonzero_start_gives_the_mixer_body_gradient_immediately`](../../../prefill/tests/test_gate_space_coupling.py) | Fails if a nonzero scale stops waking the mixer, for either architecture. |
| [`test_a_nonzero_start_no_longer_scores_exactly_like_the_gate_alone`](../../../prefill/tests/test_gate_space_coupling.py) | States both halves of the trade the flag makes. |

## D7 — 🟢 User-approved amendment — validation reports top-k overlap with the teacher

- **Background:**
  - Eviction depends only on the order of the scores, but the training objective is BCE, a calibration loss.
  - A monotone rescaling of the scores changes BCE while leaving every eviction decision identical, and the tokens whose fate can flip are the thin slice near the retention threshold.
  - Measured on the existing grids, four epochs of mixer training moved BCE by 0.33% while a random perturbation moved it by 14%, so the curve mostly reports injected noise.
- **Decision:** Validation logs the fraction of the teacher's top-k that the student also keeps, at 5, 10, 20 and 30 percent, alongside the unchanged BCE.
- **Plan gap or deviation:** The plan listed a ranking metric only as a recommended follow-up, out of scope.
- **Reason and tradeoff:**
  - Three earlier grids produced indistinguishable BCE curves and could not separate "the mixer does not help" from "the mixer did not train", which is the question these runs exist to answer.
  - It is a metric only, not a second objective: the loss and every gradient are untouched, so the runs stay comparable to the existing baselines.
  - The cost is one float per graph and token held during validation, and the metric is unavailable for a gate-only scorer because the trainer's validation pass has always required a mixer.
- **Status:**
  - 🟢 User-approved amendment. Dani asked for it as a W&B metric, explicitly not as an added loss.
  - Validation only, once per epoch; the training path is unchanged.

| Code reference | What this code does |
| --- | --- |
| [`topk_overlap`](../../../prefill/graph/training.py) | Ranks both sides per graph over the whole context and reports the agreement. |
| [`evaluate_context`](../../../prefill/graph/training.py) | Gathers the whole-context scores during validation and scores the ranking. |
| [`test_topk_overlap_reads_the_order_and_ignores_the_scale`](../../../prefill/tests/test_gate_space_coupling.py) | Fails if the metric follows a monotone rescaling or misses a swap across the threshold. |
| [`test_the_overlap_is_scored_over_the_whole_context_not_per_chunk`](../../../prefill/tests/test_gate_space_coupling.py) | Fails if the token microbatch split changes the reported ranking. |
