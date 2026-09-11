# Stage-2 accumulation and shuffling decisions

## D1 — Base and prerequisite

- **Plan gap or deviation:** None; implementation verified the planned base before editing.
- **Decision and effect:** Use a dedicated worktree on `origin/main` (`aded853`). Cherry-pick `77cfd4b` as separate prerequisite commit `8e9ec8d`.
- **Reason and tradeoff:** This restores the subgraph-at-a-time backward replay already used by the successful experiment, without including unrelated working-tree changes.
- **Status:** Explicitly approved in the plan.

## D2 — Internal counters and logged diagnostics

- **Plan gap or deviation:** The plan specifies separate counters but does not name the checkpoint's new field.
- **Decision and effect:** Keep `global_step` as processed questions and add `optimizer_step`. `wandb_step` counts emitted training/validation events. Per-question score-gradient norms are averaged as diagnostics; parameter-gradient norms describe the averaged update.
- **Reason and tradeoff:** This preserves the old data cursor and makes the different meanings of examples, updates, and logging events explicit. Validation continues weighting answer tokens.
- **Status:** Agent decision implementing the approved behavior.

## D3 — Validation environment

- **Plan gap or deviation:** The default local Python lacks the training dependencies.
- **Decision and effect:** Use the existing `/private/tmp/fastkvzip-summary-venv` CPU test environment (PyTorch 2.7.0, Transformers 4.51.3, W&B 0.30.0). The repository pins W&B 0.28.1; W&B network operations are replaced at the external boundary in the training tests.
- **Reason and tradeoff:** Reuse an available environment without changing dependency pins. These tests establish training mechanics, not a new cluster GPU memory measurement.
- **Status:** Agent decision; targeted validation passed.

## D4 — Stop limits and progress

- **Plan gap or deviation:** The plan specifies finishing the current batch but leaves progress display details open.
- **Decision and effect:** Round the per-invocation example limit to its existing epoch-local batch boundary and display that actual target. Do not change the batch partition when a limit crosses an epoch boundary.
- **Reason and tradeoff:** Progress reaches its displayed total and stop/resume preserves the uninterrupted trajectory. A requested limit of 1 with accumulation 8 can process 8 questions, as explicitly approved.
- **Status:** Agent decision implementing the approved stopping rule.

## D5 — Scope and compatibility

- **Plan gap or deviation:** No scope expansion was needed.
- **Decision and effect:** Keep legacy normalization inside stage 2, leaving shared checkpoint serialization intact. Average score-gradient norm diagnostics as scalars; they do not represent a norm over different questions' score tensors. Keep the Agentic selection and validation split, including their existing shared-context possibility.
- **Reason and tradeoff:** This implements the requested two features without changing the validation question set or introducing context-balanced sampling. New features are configurable for new runs, while exact resume inherits checkpoint settings.
- **Status:** Agent decision; validation-split and sampling scope was explicitly approved.

## Validation results

From the feature worktree's `prefill/` directory:

```bash
/private/tmp/fastkvzip-summary-venv/bin/python -m pytest -q \
  tests/test_answer_batching.py tests/test_answer_training.py \
  tests/test_graph_answer_train_cli.py tests/test_graph_training.py \
  tests/test_graph_train_cli.py tests/test_graph_model.py tests/test_graph_eval.py \
  tests/test_answer_training_slurm.py tests/test_data_load.py \
  tests/test_agentic_generation_revision.py
```

Result: **218 passed**. The suite includes exact uninterrupted/resumed-state
comparisons, independent direct-gradient references, legacy checkpoint loading,
and subgraph replay memory regression coverage. `git diff --check` passed.
No cluster jobs or GPU pilots were submitted.

## Final review

Two independent reviews examined the same implementation against the approved
plan: one focused on gradient math, schedules, and checkpoint continuation;
the other on shuffling, compatibility, metrics, documentation, and plan coverage.
Both reported no actionable correctness findings or material decision-record
omissions. No functional changes followed the 218-test validation run.
