# Stage-2 accumulation and shuffling decisions

## D1 — Base and prerequisite

- **Plan gap or deviation:** None; implementation verified the planned base before editing.
- **Decision and effect:** Use a dedicated worktree on `origin/main` (`aded853`). Cherry-pick `77cfd4b` as separate prerequisite commit `8e9ec8d`.
- **Reason and tradeoff:** This restores the subgraph-at-a-time backward replay already used by the successful experiment, without including unrelated working-tree changes.
- **Status:** Explicitly approved in the plan.

## D2 — One optimizer step

- **Plan gap or deviation:** The user superseded the original separate example/update/log-event counters and explicitly approved one optimizer step.
- **Decision and effect:** Keep only `optimizer_step` as the training clock. Derive processed questions from `epoch * training_questions + offset`. Log training and due validation together once per update. Set the default axis for automatically generated W&B charts to `train/optimizer_step` using `define_metric`.
- **Reason and tradeoff:** Schedules, cadence, checkpoints, and charts share one meaning of step. W&B manages its own history indexing; it is not a counter in the trainer or checkpoint. Letting W&B append history also avoids dropping resumed legacy metrics when old validation rows advanced its internal history index beyond the optimizer count. Existing manual chart settings are not overwritten, and legacy historical rows without `train/optimizer_step` are not backfilled.
- **Compatibility:** Legacy checkpoints map their old `global_step` to `optimizer_step`; checkpoints already containing `optimizer_step` retain it. Discard obsolete `global_step` and `wandb_step` fields in memory.
- **Status:** User-approved replacement of the original D2. W&B custom-axis mechanism is an agent implementation choice, following [W&B documentation](https://docs.wandb.ai/models/track/log/customize-logging-axes).

## D3 — Validation environment

- **Plan gap or deviation:** The default local Python lacks the training dependencies.
- **Decision and effect:** Use the existing `/private/tmp/fastkvzip-summary-venv` CPU test environment (PyTorch 2.7.0, Transformers 4.51.3, W&B 0.30.0). The repository pins W&B 0.28.1; W&B network operations are replaced at the external boundary in the training tests.
- **Reason and tradeoff:** Reuse an available environment without changing dependency pins. These tests establish training mechanics, not a new cluster GPU memory measurement.
- **Status:** Agent decision; targeted validation passed.

## D4 — Dataset size and epochs determine run length

- **Plan gap or deviation:** The user explicitly requested removing `--max-contexts` from stage 2. This supersedes both the original round-up stopping rule and the subsequent exact-cap implementation.
- **Decision and effect:** Remove the stage-2 CLI flag, option field, and per-invocation stopping logic. Use `--train-context-count` to select question-context pairs before validation holdout, and `--epochs` to determine repetitions. Unknown `--max-contexts` arguments now produce the normal argparse error. Stage 1 retains its existing flag and behavior.
- **Partial batches:** Flush only at epoch ends, using the actual question count as the denominator. Ten training questions with accumulation `8` produce `8, 2`; five questions over two epochs with accumulation `2` produce `2, 2, 1, 2, 2, 1`.
- **Resume and schedules:** Resume from completed saved updates and repeat any work performed after that checkpoint. This preserves the original batch sequence and `epochs * ceil(training_questions / K)` update horizon. Remove the special cosine-clamping option introduced solely to handle extra updates from forced partial stops. The shared scheduler implementation is restored to its pre-amendment behavior.
- **Reason and tradeoff:** One dataset-size control avoids overlapping limits and special stopping/scheduling rules. Smaller pilots use fewer selected examples or epochs; a pilot with a different dataset or horizon is a separate run, not an exact-resume configuration change.
- **Validation:** Replace cap-based resume tests with simulated interruptions immediately after actual checkpoint saves. Keep exact comparisons of resumed parameters, optimizer/scheduler state, RNG, order, counters, and metrics, including epoch-boundary checkpoints. Test rejection of the removed flag and normalized `8, 2` epoch-end updates.
- **External comparison:** [TRL SFTTrainer](https://github.com/huggingface/trl/blob/main/trl/trainer/sft_trainer.py) inherits the [Transformers training loop](https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py), which also handles a smaller final accumulation group at epoch end. This comparison does not claim identical answer-loss weighting.
- **Status:** Removal explicitly requested by the user. Test interruption mechanism is an agent implementation choice.

## D5 — Scope and compatibility

- **Plan gap or deviation:** No scope expansion was needed.
- **Decision and effect:** Keep legacy normalization inside stage 2, leaving shared checkpoint serialization intact. Average score-gradient norm diagnostics as scalars; they do not represent a norm over different questions' score tensors. Keep the Agentic selection and validation split, including their existing shared-context possibility.
- **Reason and tradeoff:** This implements the requested two features without changing the validation question set or introducing context-balanced sampling. New features are configurable for new runs, while exact resume inherits checkpoint settings.
- **Status:** User accepted D5; validation-split and sampling scope was explicitly approved.

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

Result: **220 passed** after removing the stage-2 cap. Coverage includes exact
checkpoint-resume state and metric comparisons at every tested update boundary,
both retention modes, epoch-end partial batches, legacy checkpoint loading,
independent direct-gradient references, rejected removed CLI arguments, and
subgraph replay memory regression coverage. `git diff --check` passed. No cluster
jobs or GPU pilots were submitted.

## Final review

Two independent reviews checked removal completeness, schedules, checkpoint
resume, CLI behavior, and documentation. They found one stale stage-2 pilot
example in `docs/graph-fastkvzip-experiments.md`; it now uses dataset size and
epochs and distinguishes interrupted resume from weights-only initialization.
No remaining code findings were reported. The final edit after the 220-test run
was documentation only; `git diff --check` passed again.
