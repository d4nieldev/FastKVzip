# Resume and Identity

## Identity model

Separate logical identity from execution attempts:

```text
dashboard project
└── logical grid row / W&B run
    ├── training attempt 1 + dependent evaluation attempt 1
    ├── training attempt 2 + dependent evaluation attempt 2
    └── evaluation-only retry after the final successful training attempt
```

Retries retain the original experiment name, dashboard project ID, logical run
name, W&B entity/project/run ID, checkpoint directory, and compatible evaluation
directory. New Slurm IDs are attempt history and are appended to the same
dashboard project.

Before moving an attempt to another Unix user, copy or expose the complete
checkpoint directory (including `wandb_run_id.txt`), compatible partial result
directory, and grid receipt. Verify the destination's W&B credentials can write
to the same entity/project/run. Never silently create a replacement W&B run.

## Reconcile writers first

For the logical run name on every involved account, inspect live `squeue`,
time-bounded `sacct`, `scontrol`, and output logs. Also inspect checkpoint/result
artifacts. Do not start a new writer while an existing job can write the same
checkpoint or result directory.

Slurm `COMPLETED` alone does not prove application completion; require exit
`0:0` and valid durable artifacts. W&B history is reporting evidence, not a
training checkpoint.

## Training state table

| Observed state | Action |
|---|---|
| One matching training job pending/running | Leave it; do not duplicate it. |
| Valid `last.pt` cursor reached target epochs | Training is complete; ensure evaluation finishes. |
| Valid incomplete `last.pt` | Resume training from it. |
| No valid checkpoint | Restart weights from the beginning under the same logical identity. |
| Checkpoint/config mismatch, corrupt checkpoint, or multiple possible writers | Stop and preserve evidence. |

Validate that `last.pt` belongs to the intended model/configuration and includes
model, optimizer, scheduler where configured, data cursor, RNG, prefix, prefill
chunk, and W&B run state. The data cursor, not file mtime, establishes progress.

A resume submission:

- uses the same logical run name and output directory;
- passes `--resume <run-directory>/last.pt`;
- passes `--epochs <original-total-target>`—not epochs remaining;
- never combines `--resume` with `--gate-checkpoint`;
- normally omits a prior pilot's `--max-contexts`; and
- preserves the original W&B entity/project and checkpoint run ID.

`wandb_run_id.txt` and the checkpoint's W&B ID must agree when both exist. With
no checkpoint, preserve `wandb_run_id.txt` so a from-zero retry still reports to
the same W&B run. If prior metrics exist, report that weights are restarting;
do not hide the failed attempt by creating a new run.

### OOM recovery

First determine the failing phase and peak. Current checkpoints treat token and
graph microbatches as resume invariants. If a valid intermediate checkpoint
exists, keep its microbatches and move to a larger compatible GPU when possible.
If the only safe fix is a smaller microbatch, do not alter checkpoint metadata:
obtain approval to restart weights under the same W&B/dashboard identity, or
stop until the pipeline supports changing that invariant.

## Pair replacement attempts

Whenever training is resubmitted, immediately submit a new evaluation attempt
with both:

```text
--dependency=afterok:<replacement-training-id>
--existing-results resume
```

The failed training and its unsatisfied evaluation remain recorded. The new
pair uses the same logical run/evaluation names and dashboard project. Before
submitting, prove no earlier evaluator can still write the result directory.

## Evaluation state table

Every evaluation invocation includes `--existing-results resume`, even when its
directory does not exist.

| Observed state | Action |
|---|---|
| Matching evaluation job pending/running | Leave it; do not duplicate it. |
| Complete compatible outputs and metrics | Do not recompute. |
| Compatible partial outputs | Resume the same evaluation directory. |
| Evaluation failed before writing outputs | Resume the same directory/name. |
| Manifest mismatch/corruption or overlapping writer | Stop; never overwrite. |

Every evaluation attempt, including an evaluation-only retry, uses
`afterok:<final-successful-training-id>`. If Slurm can no longer resolve that
historical dependency, stop and report the constraint rather than silently
dropping the relationship.

A compatible result manifest must match the resolved checkpoint path, training
W&B run ID, protected-window size, pruning level, and prefill mode. Resume skips
saved task/example/ratio work and may add missing full-cache answers. Only one
job may write a result directory.

Evaluation must start only after the final training target succeeds. `last.pt`
is replaced in place, while the result manifest does not hash its bytes. If the
checkpoint changed after partial evaluation, do not combine old and new outputs;
preserve them and request a new evaluation-identity decision.

## Dashboard on retry

For every retry batch, POST `/api/projects` again with the exact original
experiment name. Read the returned ID and require it to equal the receipt's
dashboard project ID. POST all newly returned train/evaluation IDs together to
that project's jobs endpoint and verify `assigned` equals the submitted count.

Never create a suffix such as `-retry`, `-resume`, or `-oom` for the W&B run or
dashboard project. Attempt-specific Slurm log names and job IDs provide the
history without fragmenting the experiment.
