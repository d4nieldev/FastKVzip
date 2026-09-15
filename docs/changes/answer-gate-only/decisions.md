# Gate-only answer training decisions

## D1 — `graph_dim = None` is the only record of "no mixer"

- **Plan gap or deviation:** The plan said the checkpoint records the mode as a single
  fact but did not fix which field carries it.
- **Decision and effect:** `graph_dim` becomes `int | None` end to end — the
  `ImplicitGraphScorer` argument, `AnswerTrainingOptions`, and the saved config. `None`
  means no mixer; `0` is still rejected.
- **Reason and tradeoff:**
  - A separate boolean would be a second fact that can disagree with `graph_dim`, and it
    would have to be added to `normalized_answer_resume_config` for every existing
    checkpoint to keep resuming.
  - `graph_dim` is already in `_CONFIG_KEYS`
    ([`prefill/graph/evaluation.py`](../../../prefill/graph/evaluation.py)), so a
    gate-only checkpoint keeps every historical key present and only changes one value.
  - Cost: `graph_dim` is no longer unconditionally an int, so its two readers needed a
    branch.
- **Coverage:** `test_gate_only_checkpoint_validates_and_must_carry_no_mixer` asserts
  `graph_dim: None` validates, `graph_dim: 0` still raises "positive integer", and a
  stray mixer tensor is still rejected.
- **Status:** Agent decision filling a plan gap.

## D2 — The gate reads the master dtype, so dropping the mixer is not also a precision change

- **Plan gap or deviation:** The plan required gate-only training to stay in fp32 but did
  not say how, given that nothing in the code chose fp32 deliberately.
- **Decision and effect:** A new `ImplicitGraphScorer.hidden_dtype`
  ([`prefill/graph/model.py`](../../../prefill/graph/model.py)) returns `compute_dtype`
  when a mixer exists and the master dtype when one does not. The scoring paths
  materialize hidden states in it.
- **Reason and tradeoff:**
  - With a mixer, the delta is accumulated in the master dtype and promotes `hidden +
    delta` to fp32. That promotion is incidental, but it is the precision every existing
    measurement was taken at.
  - Removing the mixer removes the promotion. Scoring would silently fall to bf16. I
    measured that gap directly on this code at 4096 hidden, 16 sinks, 5 query groups and
    2000 tokens: **1.5e-3 mean relative score difference, 8.9e-4 max absolute**, and
    **0.44% of the retained set changing at 10% retention** (0.21% at 30%). The plan
    quoted ~1% churn from a review measurement; the exact churn depends on the weights,
    but the order of magnitude holds, and it is the same order as the effect the ablation
    is looking for.
  - Mixer runs are unaffected: the property returns exactly what the code passed before.
  - Cost: gate-only scoring uses fp32 activations, so it is not as cheap as bf16 would be.
    Correct comparison is worth more than that memory.
- **Coverage:** `test_gate_only_scorer_holds_the_mixer_arms_precision` builds a bf16
  scorer and asserts the scores come back fp32.
- **Status:** Agent decision filling a plan gap.

## D3 — No delta at all, rather than a zero delta

- **Plan gap or deviation:** The plan said gate-only is "delta is zero" without fixing
  whether a zero tensor is built.
- **Decision and effect:** `_HeadwiseGateAdapter.forward_batch` takes `delta: Tensor |
  None` and uses `mixed = hidden` when it is `None`. `prepare` returns `None`, and
  `score_prepared` raises if a prepared state and the mixer disagree about existing.
- **Reason and tradeoff:**
  - A zero delta would allocate a second `[graphs, tokens, hidden_dim]` tensor — one of
    the largest allocations in the step — to add nothing.
  - It would also have promoted the dtype, hiding D2 instead of deciding it.
  - The paired `None` check is a guard rather than a silent fallback, because passing a
    prepared state to a gate-only scorer would otherwise be ignored rather than caught.
- **Coverage:** `test_gate_only_scorer_matches_the_plain_gate_forward`,
  `test_gate_only_scorer_rejects_a_prepared_mixer_state`, and
  `test_gate_only_scoring_runs_none_of_the_mixer_math`, which stubs
  `ImplicitGraphMixer.prepare_from_chunks` and `.delta` and asserts neither is reached.
- **Status:** Agent decision filling a plan gap.

## D4 — Mixer metrics are logged as `0.0` rather than removed from the allowlist

- **Plan gap or deviation:** The plan proposed subtracting a `_MIXER_LOG_KEYS` set from
  the W&B allowlist. That was not implemented.
- **Decision and effect:** `TRAIN_LOG_KEYS` and its KL twin are untouched.
  `train/mean_alpha`, `train/mixer_learning_rate` and `train/mixer_grad_norm` report
  `0.0` in a gate-only run.
- **Reason and tradeoff:**
  - Subtraction would have multiplied against the existing loss twin: loss × mixer is four
    key sets and a two-dimensional conditional at each assertion.
  - All three values are true, not placeholders. A gate-only model *is* the `alpha = 0`
    limit of a mixer model, its mixer gradient is zero, and no mixer learning rate is
    applied.
  - Keeping the schema identical is what lets a gate-only run and a mixer run be charted
    on the same W&B axes, which is the entire purpose of the ablation. This follows D4 of
    the KL change ("history stays comparable with completed grids") more closely than the
    planned subtraction did.
  - Cost: three permanently-zero series in a gate-only run.
- **Coverage:** `test_gate_only_run_trains_the_gate_alone_and_saves_no_mixer` asserts a
  gate-only run's emitted keys are exactly `TRAIN_LOG_KEYS | VALIDATION_LOG_KEYS`, the
  same set a mixer run emits, and that the three mixer values are `0.0`.
- **Status:** Agent decision; deviation from the planned edit.

## D5 — `--gate-checkpoint` does not become a fourth initialization mode

- **Plan gap or deviation:** The plan put `--gate-checkpoint` in the mutually exclusive
  source group but did not say whether `initialization` gains a value.
- **Decision and effect:** A `--gate-checkpoint` run stays `initialization == "fresh"`.
  Provenance lives only in `options.gate_checkpoint`.
- **Reason and tradeoff:**
  - Everything `initialization` controls is already correct for a fresh run: no payload is
    required, the architecture is not strict, the cursor and RNG start clean, and a new
    W&B run is created. A bare gate file dictates nothing else.
  - `restore_training_state` needed no change at all as a result.
  - Cost: the checkpoint does not record which gate file seeded the run. Recording it
    would make `_validate_resume_config` fail every resume unless it were also threaded
    back out of the saved config, for provenance the run directory and W&B name already
    carry.
- **Coverage:** `test_the_three_initialization_sources_are_mutually_exclusive` and
  `test_a_gate_file_selects_the_gate_shape_and_conflicts_are_rejected`.
- **Status:** Agent decision filling a plan gap.

## D6 — A local gate file resolves the gate shape before the model loads

- **Plan gap or deviation:** The plan said `--gate-checkpoint` would reuse stage 1's
  `_student_gates`. That helper only handles the literal string `"fastkvzip"`; a path
  falls through to random gates with no warning.
- **Decision and effect:** `resolve_options` takes a third `gate_payload` argument and
  reads `train_graph._checkpoint_gate_metadata` from it, which supplies `gate_dim`,
  `gate_sink` and `compute_dtype` and raises on a contradictory flag.
  `_make_components` then calls `load_gate_checkpoint` after the scorer exists. This is
  the same two-site structure stage 1 uses.
- **Reason and tradeoff:**
  - Reusing `_student_gates` alone would have made `--gate-checkpoint <path>` silently
    train from random init — the failure mode is a wasted multi-hour run, not an error.
  - Resolving the metadata before the LLM loads makes a wrong `--gate-dim` fail in a
    second rather than after an 8B model is in memory.
  - The third parameter is defaulted, so the existing two-argument `resolve_options` calls
    throughout the tests are unchanged.
- **Coverage:** `test_a_gate_file_selects_the_gate_shape_and_conflicts_are_rejected`.
- **Status:** Agent decision; the plan's stated approach was incomplete.

## D7 — `-g` decides it is a path by shape, not by existence

- **Plan gap or deviation:** The plan said the gate loader checks "whether the argument is
  an existing file". Deciding by existence makes a mistyped path silently fall back.
- **Decision and effect:** `is_gate_path`
  ([`prefill/attention/gate.py`](../../../prefill/attention/gate.py)) is true for a value
  ending in `.pt` or containing a separator. A path-shaped value that does not exist
  raises `FileNotFoundError`.
- **Reason and tradeoff:**
  - A released gate is always a bare stem, so the rule cannot capture one.
  - Deciding by existence would send a typo to the hub, then to the broken
    `~/FastKVzip/result_gate` fallback, and finally to a bare `KeyError`.
  - The same predicate is what `prefill/args.py` uses to skip the eviction-structure
    substring match, so the two cannot drift apart.
- **Coverage:** `test_a_trained_gate_reloads_through_the_evaluator_and_reproduces_its_scores`.
- **Status:** Agent decision; refinement of the planned rule.

## D8 — Sink comes from the weights for released gates too

- **Plan gap or deviation:** The plan said sink inference from `k_base` was needed for
  local paths, leaving open whether the filename regex stays for released names.
- **Decision and effect:** The `sink(\d+)` regex is deleted. Every gate reads
  `k_base.shape[-2]`, and a sink below 1 is rejected.
- **Reason and tradeoff:**
  - The two always agree for a released gate, whose name encodes the same number its
    tensor has, so nothing changes for the existing path.
  - Keeping both would leave two sources of truth and a dead branch.
  - The `sink < 1` check matters: an empty sink axis makes the score `1/(1+0) = 1` for
    every token, so the whole gate would silently degrade to tie-break order rather than
    fail.
- **Coverage:** `test_a_trained_gate_reloads_through_the_evaluator_and_reproduces_its_scores`
  asserts the reloaded sink.
- **Status:** Agent decision filling a plan gap.

## D9 — `Weight.forward` casts its input to the gate dtype

- **Plan gap or deviation:** The plan said "the evaluator will score a gate in the
  precision that gate was saved in" without saying where the cast happens.
- **Decision and effect:** One cast at the top of `Weight.forward`.
- **Reason and tradeoff:**
  - Training saves fp32 master weights, so without a cast the first projection raises on
    bf16 hidden states — after the model has loaded.
  - Casting the *gate* down to the model dtype instead would run, but would reintroduce
    exactly the bf16 drift D2 exists to avoid, so a fine-tuned gate could not reproduce
    its own validation scores.
  - It is a no-op for a released gate, whose projections already carry the model dtype.
- **Coverage:** `test_a_trained_gate_reloads_through_the_evaluator_and_reproduces_its_scores`
  compares reloaded scores against the scorer's own output.
- **Status:** Agent decision filling a plan gap.

## D10 — Layer regrouping sorts numerically and asserts the full range

- **Plan gap or deviation:** The plan listed layer ordering as a trap; this records the
  remedy.
- **Decision and effect:** `_layer_state_dicts` parses each key's leading index as an
  integer and raises unless the indices are exactly `0..n-1`.
- **Reason and tradeoff:**
  - A checkpoint's gate keys are `"0."…"35."` as text. Sorting them as text gives
    0, 1, 10, 11, 2 and assigns each layer's gate to the wrong layer.
  - Every layer has identical tensor shapes, so `load_state_dict(strict=True)` and the
    checkpoint validator both accept the permutation. The only symptom is scores that are
    worse than expected.
  - The range assertion catches a truncated or padded file, which the sort alone would
    not.
- **Coverage:** `test_layer_order_survives_a_checkpoint_with_ten_or_more_layers` uses
  twelve layers, which is the smallest count where text and numeric order differ.
- **Status:** Agent decision filling a plan gap.

## D11 — `load_gate` passes the model's device instead of defaulting to `cuda`

- **Plan gap or deviation:** Not mentioned in the plan; found while making the loader
  testable.
- **Decision and effect:** `load_gate` now calls `load_fastkvzip(..., device=model.device)`.
  The parameter default is unchanged for other callers.
- **Reason and tradeoff:**
  - The default was the literal string `"cuda"`, so the gate always landed on `cuda:0`
    regardless of which device the model was on.
  - `ModelKVzip` sets `self.device` before it calls `load_gate`, so the value is available.
  - It is also what lets the new loader tests run on CPU at all.
- **Coverage:** Every test in `tests/test_gate_only_scoring.py` loads with `device="cpu"`.
- **Status:** Agent decision; a fix adjacent to the requested change, kept because the
  change cannot be tested without it.

## D12 — The batching fixture takes a `graph_mixer` keyword rather than reading the flags

- **Plan gap or deviation:** The plan called for a `batch_run` case but not how the shared
  fixture, which hardcodes `--graph-dim 2` and a mixer LR scheduler, would accommodate one.
- **Decision and effect:** `run(...)` gains `graph_mixer=True`; when false it omits those
  defaults. It does not infer the mode from the presence of `--no-graph-mixer` in the
  flags.
- **Reason and tradeoff:**
  - Inferring from the flags would have made the resume case impossible to write: a
    gate-only resume deliberately does *not* repeat `--no-graph-mixer`, so the fixture
    would have re-added `--graph-dim`, which that run correctly rejects.
  - Mixer runs keep the exact argument list they had, so every existing test in the file
    is unaffected.
- **Coverage:** `test_gate_only_run_resumes_without_repeating_the_flag` is the case that
  forced it.
- **Status:** Agent decision filling a plan gap in test design.

## D13 — Stage-1 `GraphTrainer` keeps its mixer assumptions

- **Plan gap or deviation:** The plan scoped the mode to answer training. This records
  what was deliberately *not* changed.
- **Decision and effect:** `GraphTrainer` in
  [`prefill/graph/training.py`](../../../prefill/graph/training.py) still dereferences
  `scorer.mixer` in four places. Only `build_adamw_optimizers` and
  `_model_gradient_norms`, which answer training shares, were made mixer-optional.
- **Reason and tradeoff:**
  - `train_graph.py` has no `--no-graph-mixer`, so it can never construct a mixer-free
    scorer, and the guards would be unreachable code.
  - Gate-only distillation would reproduce FastKVzip's own training, so there is no
    experiment waiting on it.
  - Cost: adding the mode to stage 1 later means revisiting those four sites.
- **Status:** Agent decision; a deliberate non-change.

## Validation results

`cd prefill && python -m pytest tests/ -q` → **737 passed**, including 18 new tests. An
intermediate run had exactly two failures, both in the new end-to-end tests and both my
own test bugs (a log-union that also caught validation keys, and a stop hook that needs
`--save-strategy steps`); every pre-existing test passed at that point and still does.

A mixer run is **bit-identical to `HEAD`**. Loading the pre-change `graph/model.py`
alongside the new one, giving both scorers the same weights, and comparing
`ImplicitGraphScorer.forward` at two token-microbatch sizes and `score_subgraph_batch`
across every graph batch gives `torch.equal` in every case, with the same `state_dict`
keys. This is the evidence that `hidden_dtype` changed nothing for existing runs.

The released-gate load path is likewise bit-identical. Loading an upstream-format
`q5_dim16_sink16.pt` through both the pre-change and the new `load_fastkvzip` gives the
same inferred sink, the same dtype, and `torch.equal` scores on every layer — so reading
the sink from `k_base` and casting in `Weight.forward` changed nothing for a released
gate.

Mutation checks on the two findings that would fail silently rather than loudly:

- Reverting `hidden_dtype` so a gate-only scorer uses `compute_dtype` fails
  `test_gate_only_scorer_holds_the_mixer_arms_precision` and
  `test_a_trained_gate_reloads_through_the_evaluator_and_reproduces_its_scores`.
- Sorting the checkpoint's layer keys as text instead of as integers fails
  `test_layer_order_survives_a_checkpoint_with_ten_or_more_layers`.

### Limitations

- **The suite ran on torch 2.14.0 and transformers 4.51.3, not the pinned torch 2.7.0**
  ([`prefill/requirements.txt`](../../../prefill/requirements.txt)); torch 2.7.0 has no
  macOS arm64 wheel for this Python. `flash-attn` is CUDA-only and was not installed, so
  nothing that imports it was exercised — the full suite collects and passes without it.
  Re-run on the cluster venv before submitting a grid.
- **Nothing here ran on a GPU or on a real model.** Every test uses toy dimensions and a
  stub teacher. The three cluster checks in the plan — a 3-context pilot, `eval_chunk.py`
  on a fine-tuned gate against `-g fastkvzip`, and a zero-step checkpoint that must
  reproduce `-g fastkvzip` exactly — are all still outstanding, and the third is the one
  that proves the save/load/precision round trip end to end on real weights.
- **The released FastKVzip gate was never downloaded.** `--gate-checkpoint fastkvzip`
  goes through stage 1's existing `_student_gates`, which is exercised by
  `train_graph.py`'s own tests, but the answer-training path to it is covered only by the
  local-file case.
- Memory and step time for this mode are unmeasured. The `AGENTS.md` scorer-width table
  was taken with a mixer and does not apply.
