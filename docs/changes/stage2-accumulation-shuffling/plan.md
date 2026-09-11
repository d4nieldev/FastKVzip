# Stage-2 gradient accumulation and training-data shuffling

## Summary

Add configurable accumulation across questions and enable shuffling by default for new stage-2 runs. Preserve equal weight per question and reproducible checkpoint resume.

Create a dedicated worktree and feature branch from current `origin/main`—verified at `aded853`. Include the existing replay-memory fix, `77cfd4b`, as a separate prerequisite commit in the same PR.

## Training behavior

- Add `--gradient-accumulation-steps K`, a positive integer defaulting to **1**.
- Process questions sequentially, retaining parameter gradients across the batch. Average gradients by the actual number of questions, then step both optimizers once. Do not automatically scale learning rates.
- Finish each epoch with a correctly normalized partial batch. For five questions and `K=2`, updates use batches of **2, 2, 1**.
- Add `--shuffle-data` / `--no-shuffle-data`, defaulting to **enabled for new runs**, including weights-only initialization.
- Shuffle individual training question–context pairs each epoch using a separate `random.Random(seed + epoch)` instance. Reconstruct that permutation on resume and interpret the saved offset within it.
- Keep data selection, validation membership, and token-weighted validation metrics unchanged. Context-balanced sampling and context-disjoint validation are outside this PR.

## Schedules, logging, and checkpoints

- Keep the existing example counter and add an optimizer-update counter. The update horizon is `epochs × ceil(training_examples / K)`.
- Advance learning-rate schedulers and step-based save/evaluation cadence per optimizer update. Epoch-based cadence continues to follow completed epochs.
- Sample uniform retention independently per question. Linear retention follows the optimizer-update horizon, sharing one ratio throughout each accumulated batch.
- Preserve the cosine scheduler’s requirement for at least two updates; report a clear error for an insufficient horizon.
- Log training NLL and accuracy once per update as equal-question averages. Average per-question score-gradient diagnostics; measure gate/mixer gradient norms after gradient averaging. Add `train/examples`, `train/optimizer_step`, and `train/batch_examples`.
- Save and evaluate only after completed updates. Honor `--max-contexts` after finishing the current batch, processing at most `K−1` extra questions without crossing the total training target. Do not serialize pending gradients.
- Store both new options in checkpoint configuration. Normalize legacy stage-2 checkpoints in memory to accumulation **1**, shuffling **disabled**, and an update counter equal to their existing example counter before strict compatibility checks.
- Exact resume retains saved settings and rejects conflicting overrides. Weights-only initialization may select new settings.

## Validation

Extend the existing CPU toy-model tests using real scorer, replay, optimizer, and training-loop behavior:

- Accumulated updates match the explicit average of per-question losses, including unequal answer lengths.
- Cover `K=1`, `K` larger than an epoch, partial batches, frozen LLM weights, and optimizer/scheduler counts.
- Verify deterministic shuffling visits every training example once per epoch, excludes validation examples, and leaves the retention RNG stream unchanged.
- Compare uninterrupted training with stop/resume: example order, retention draws, parameters, optimizer/scheduler state, counters, and logged metrics.
- Cover epoch-boundary stopping: five questions, `K=2`, two epochs must retain the batch sequence **2, 2, 1, 2, 2, 1** after resume.
- Test legacy checkpoints, incompatible resume overrides, weights-only initialization, batch metrics, save/evaluation cadence, and Slurm argument forwarding.
- Run the existing replay-memory regression tests and relevant training/checkpoint tests.

## Delivery

Update the stage-2 documentation with accumulation, shuffle defaults, metric meanings, and stopping behavior. Preserve unrelated working-tree changes.

At implementation start, save the approved plan and record subsequent decisions under `docs/changes/stage2-accumulation-shuffling/`. Commit both documents with the feature changes, push the branch, and open one PR containing the prerequisite fix and both features. Cluster submission is a separate task.
