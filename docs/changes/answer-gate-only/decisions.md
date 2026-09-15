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

## D4 — `--no-graph-mixer` will not strip the mixer off an existing checkpoint

- **Plan gap or deviation:** The plan calls the two flags "independent" and names only one
  exclusivity rule, between the three initialization sources. It does not say that
  `--graph-checkpoint … --no-graph-mixer` is refused.
- **Decision and effect:** When the checkpoint being resumed or warm-started from has a
  mixer, passing `--no-graph-mixer` raises and points at `--gate-checkpoint`. A gate-only
  checkpoint infers the mode without the flag, so nothing else is restricted.
- **Reason and tradeoff:**
  - It looks like the sharpest form of the ablation, but it is a different experiment: the
    gate in a stage-1 checkpoint was trained *alongside* a mixer and is co-adapted to it.
    Fine-tuning that gate alone measures "what happens when you remove a mixer the gate
    expects", not "does the mixer earn its keep over the released gate".
  - The user ruled this initialization source out when choosing between the options, so
    silently allowing it would contradict a decision already made.
  - Cost: if that experiment is ever wanted, it needs a flag of its own. The error names
    the supported path rather than just refusing.
- **Coverage:** `test_no_graph_mixer_cannot_strip_the_mixer_off_an_existing_checkpoint`.
- **Status:** Agent decision filling a plan gap.

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
- **Coverage:** `test_a_gate_file_selects_the_gate_shape_and_conflicts_are_rejected` calls
  `resolve_options` with no checkpoint payload, which only succeeds on the `fresh` path —
  the other modes raise "checkpoint payload is required". Its sibling
  `test_the_three_initialization_sources_are_mutually_exclusive` pins the argparse rule.
- **Status:** Agent decision filling a plan gap.

## D6 — A local gate file resolves the gate shape before the model loads

- **Plan gap or deviation:** The plan said `--gate-checkpoint` would reuse stage 1's
  `_student_gates`. That helper only handles the literal string `"fastkvzip"`; a path
  falls through to random gates with no warning.
- **Decision and effect:** `run_training` loads a local gate file and passes it to
  `resolve_options`, which reads `train_graph._checkpoint_gate_metadata` for `gate_dim`,
  `gate_sink` and `compute_dtype` and raises on a contradictory flag. `_make_components`
  then calls `load_gate_checkpoint` after the scorer exists. This is the same two-site
  structure stage 1 uses.
- **Reason and tradeoff:**
  - Reusing `_student_gates` alone would have made `--gate-checkpoint <path>` silently
    train from random init — the failure mode is a wasted multi-hour run, not an error.
  - Resolving the metadata before the LLM loads makes a wrong `--gate-dim` fail in a
    second rather than after an 8B model is in memory.
  - The `gate_payload` parameter is defaulted, so the existing two-argument
    `resolve_options` calls throughout the tests are unchanged.
- **Coverage:** `test_a_gate_file_conflicting_with_gate_dim_fails_before_the_model_loads`
  drives the real `run_training` and asserts the model factory is never called — reverting
  the `run_training` wiring alone makes it fail, which is how the missing wiring was
  caught. `test_gate_checkpoint_weights_actually_reach_the_scorer` then proves the weights
  land in the scorer rather than being silently ignored, and
  `test_a_gate_file_selects_the_gate_shape_and_conflicts_are_rejected` covers the
  resolution rules.
- **Status:** Agent decision; the plan's stated approach was incomplete.

## D7 — `-g` decides it is a path by shape, not by existence

- **Plan gap or deviation:** The plan required `-g` to accept a path but not how a path is
  told apart from a released gate name.
- **Decision and effect:** `is_gate_path` is true for a value ending in `.pt` or
  containing a separator. A path-shaped value that does not exist raises
  `FileNotFoundError` rather than falling back to the hub.
- **Reason and tradeoff:**
  - A released gate is always a bare stem, so the rule cannot capture one.
  - Deciding by existence would send a typo to the hub, then to the broken
    `~/FastKVzip/result_gate` fallback, and finally to a bare `KeyError`.
  - The same predicate is what `prefill/args.py` uses to skip the eviction-structure
    substring match, so the two cannot drift apart.
- **Coverage:** `tests/test_gate_path_arguments.py` — the missing-path case, the released
  names keeping their exact level and tag, and a `snapshots/` path not being read as
  `snap`.
- **Status:** Agent decision filling a plan gap.

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
- **Coverage:** `test_a_gate_file_without_a_sink_is_rejected`, and the released-gate
  comparison in the validation section below.
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

## D10 — The results tag is `<run-dir>-<file-stem>`, not the run directory alone

- **Plan gap or deviation:** The plan said "the run directory name goes into the tag
  instead". The stem is appended as well.
- **Decision and effect:** A path-valued `-g` tags results `_<parent-dir>-<stem>`; a
  released name keeps its existing `_<name>` tag exactly.
- **Reason and tradeoff:**
  - The run directory alone would merge `best.pt` and `last.pt` from one run into the same
    results directory, which is the same collision the plan set out to remove, one level
    down.
  - The tag is the string that names every result file and the evaluation run's identity,
    so it has to distinguish everything that can differ.
  - Cost: longer directory names.
- **Coverage:** `test_result_tags_distinguish_runs_whose_checkpoints_share_a_name` checks
  two runs and two checkpoints of one run all differ.
- **Status:** Agent decision; deviation from the plan's wording.

## D11 — Inert mixer settings stay in the config at their defaults

- **Plan gap or deviation:** The plan said the mixer's knobs "fail fast rather than sit
  meaninglessly in the checkpoint and the W&B config". Explicitly passed flags do fail
  fast, but their *defaults* are still written.
- **Decision and effect:** A gate-only checkpoint records `graph_dim: null` alongside
  `gram_normalization: "token-count"`, `leaky_relu_slope: 0.01`, `alpha_init: 0.1` and
  `mixer_lr: 0.001`, none of which are applied.
- **Reason and tradeoff:**
  - `_CONFIG_KEYS` is a fixed tuple that every checkpoint must satisfy, so dropping those
    keys would make gate-only checkpoints unloadable rather than tidier.
  - Because the matching flags are rejected, a recorded value is always the untouched
    default — never a number someone chose and expected to matter — and `graph_dim: null`
    sits next to them saying so.
  - The alternative, writing `null` for all four, would mean branching
    `normalized_checkpoint_config`, which stage 1 shares and which this change otherwise
    does not touch.
- **Coverage:** `test_gate_only_run_trains_the_gate_alone_and_saves_no_mixer` asserts the
  saved `graph_dim` is `None`.
- **Status:** Agent decision; the plan's promise holds for flags, not for defaults.

## D12 — `load_gate` passes the model's device instead of defaulting to `cuda`

- **Plan gap or deviation:** Not mentioned in the plan; found while making the loader
  usable from a path.
- **Decision and effect:** `load_gate` calls `load_fastkvzip(..., device=model.device)`.
  The parameter default is unchanged for other callers.
- **Reason and tradeoff:**
  - The default was the literal string `"cuda"`, so the gate always landed on `cuda:0`
    regardless of which device the model was on — wrong on any non-zero device.
  - `ModelKVzip` sets `self.device` before it calls `load_gate`, so the value is there.
  - Cost: this line is not covered. `load_gate` has two callers, neither reachable from
    the CPU-only suite, and the tests exercise `load_fastkvzip` directly. Covering it
    would mean a `ModelKVzip` stub that exists for one argument.
- **Status:** Agent decision; a fix adjacent to the requested change, uncovered and
  deliberately so.

## D13 — Stage-1 `GraphTrainer` keeps its mixer assumptions

- **Plan gap or deviation:** The plan scoped the mode to answer training. This records
  what was deliberately *not* changed.
- **Decision and effect:** `GraphTrainer` in
  [`prefill/graph/training.py`](../../../prefill/graph/training.py) still dereferences
  `scorer.mixer` in five places across four methods. Only `build_adamw_optimizers` and
  `_model_gradient_norms`, which answer training shares, were made mixer-optional.
- **Reason and tradeoff:**
  - `train_graph.py` has no `--no-graph-mixer`, so it can never construct a mixer-free
    scorer, and the guards would be unreachable code.
  - Gate-only distillation would reproduce FastKVzip's own training, so there is no
    experiment waiting on it.
  - Cost: adding the mode to stage 1 later means revisiting those five sites.
- **Status:** Agent decision; a deliberate non-change.

## Validation results

`cd prefill && python -m pytest tests/ -q` → **747 passed**, including 28 new tests. An
intermediate run had exactly two failures, both in the new end-to-end tests and both my
own test bugs (a log-union that also caught validation keys, and a stop hook that needs
`--save-strategy steps`); every pre-existing test passed at that point and still does.

**A review pass caught a real bug before this shipped.** The `gate_payload` argument was
threaded into `resolve_options` but never passed by `run_training`, so
`--gate-checkpoint <path>` skipped the early shape check entirely and a contradictory
`--gate-dim` would have surfaced as a state-dict error after the teacher model was
loaded. The wiring and
`test_a_gate_file_conflicting_with_gate_dim_fails_before_the_model_loads` are the fix;
reverting the wiring alone still fails that test.

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
- Dropping the `gate_payload` wiring from `run_training` fails
  `test_a_gate_file_conflicting_with_gate_dim_fails_before_the_model_loads`.

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
