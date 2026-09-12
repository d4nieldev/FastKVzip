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

## D4 — Exact stopping and partial updates

- **Plan gap or deviation:** The user rejected the approved plan's rule allowing up to `K-1` extra questions. The original D4 rounding behavior is superseded.
- **Decision and effect:** Honor `--max-contexts` exactly per invocation. Flush and normalize the final batch by its actual size: cap `10`, accumulation `8` gives `8, 2`. Epoch ends also flush partial batches. No batch crosses an epoch or the total training target; checkpoints still contain no pending gradients.
- **Reason and tradeoff:** The limit means what it says. A forced mid-batch stop changes the update grouping and may add updates after resume. Exact equality to an uninterrupted run is guaranteed at existing batch boundaries; arbitrary-cap resume instead preserves the saved state, next question, and RNG streams, but cannot recreate a batch already flushed.
- **Schedules:** Retain the original `epochs * ceil(training_questions / K)` horizon on resume. Linear retention stays at its endpoint after this horizon. Enable endpoint clamping for stage 2 in the shared warmup/cosine scheduler so extra updates cannot make its learning rate rise again. This endpoint is zero: post-horizon updates advance optimizer moments but do not change parameters. Stage 1 retains its existing scheduler behavior; its optimizer can step multiple times per context, so a global clamp would change supported stage-1 runs. Other schedulers continue stepping normally. Changing the horizon on every pilot would alter the early learning rates and conflict with restoring the saved scheduler state.
- **Two limits:** Keep the existing distinct controls. `--train-context-count` selects question-context pairs before the validation split and determines the dataset reused each epoch. `--max-contexts` only limits this invocation's work. It does not change the dataset, validation set, or schedule horizon. Clarify this in CLI help and documentation; deleting either control was not requested.
- **External comparison:** [TRL SFTTrainer](https://github.com/huggingface/trl/blob/main/trl/trainer/sft_trainer.py) inherits the Transformers training loop. [Transformers Trainer](https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py) processes a smaller final accumulation group at epoch end and accounts for its actual size. This supports flushing incomplete groups; its `max_steps` counts optimizer updates, not examples. This comparison does not claim identical answer-loss weighting.
- **Status:** Exact stopping explicitly requested by the user. Fixed-horizon handling and retaining both distinct existing flags are agent implementation choices.

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

Result: **222 passed** after the D2/D4 amendments. Coverage includes exact
uninterrupted/resumed-state comparisons at existing batch boundaries, exact
example caps with normalized partial updates, reproducible continuation from
forced partial updates, both retention modes, legacy checkpoint loading,
independent direct-gradient references, and subgraph replay memory regression
coverage. `git diff --check` passed. No cluster jobs or GPU pilots were submitted.

## Final review

Two independent reviews checked math/schedules/resume and logging/compatibility/
plan coverage. Review found that a global cosine clamp would also change stage 1,
whose update count can exceed its context-based horizon. Clamping is now enabled
only for stage 2, with a regression covering both behaviors. Documentation now
states that post-horizon cosine LR is zero and that W&B custom axes affect default
charts without rewriting legacy rows. Final validation was rerun after these fixes.
