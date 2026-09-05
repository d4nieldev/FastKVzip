---
name: running-fastkvzip-experiments
description: Use when planning, submitting, distributing, resuming, monitoring, or evaluating FastKVzip experiment grids on the BGU Slurm cluster.
---

# Running FastKVzip Experiments

## Core contract

Treat a configured run as one logical run across every Slurm attempt. An OOM,
timeout, preemption, retry, or account move creates new job IDs—not a new W&B
run or dashboard project.

Planning and inspection are read-only. Submit, cancel, copy credentials, or
overwrite data only with the user's explicit authorization. If
`using-bgu-slurm` is available, apply it for connection, live-policy, resource,
monitoring, and cancellation details.

## Read the relevant guidance

- Read [references/microbatches.md](references/microbatches.md) whenever choosing
  or changing graph/token microbatches or GPU type.
- Read [references/runbook.md](references/runbook.md) before planning or
  submitting a grid, dividing it across users, or registering the dashboard.
- Read [references/resume-and-identity.md](references/resume-and-identity.md)
  whenever jobs or artifacts already exist, or any attempt failed/interrupted.

Read each selected reference completely before acting.

## Required workflow

1. Expand the exact Cartesian grid into a manifest. Include run name, full
   arguments, GPU/microbatches, assigned SSH identity, cache/storage mode,
   checkpoint/result paths, W&B identity, and evaluation protocol.
2. Discover users dynamically with `scripts/discover_slurm_users.py`; verify
   authorization and capabilities. Assign complete train/evaluation pairs by
   constraints and estimated GPU-hours. Never hardcode a user count.
3. Reconcile live jobs and durable artifacts. Verify one commit/environment,
   unique outputs, cache readiness, W&B access, and justified Slurm resources.
4. Show the exact manifest and submission/resource requests; obtain approval.
5. For every row, submit training once with `sbatch --parsable`, persist its ID,
   then immediately submit its evaluation with
   `--dependency=afterok:<train-id>`. Every evaluation command includes
   `--existing-results resume`, including its first attempt.
6. POST the original experiment name to the dashboard project endpoint, read
   its returned `id`, then POST every returned train/evaluation ID in one jobs
   call. Persist the project ID and verify `assigned`.
7. Monitor exact IDs. Resume/retry only after classifying scheduler state,
   checkpoints, evaluation outputs, and existing writers.

## Non-negotiable identities

| Object | Retry/resume rule |
|---|---|
| Logical training run | Same run name and output directory |
| W&B | Same entity, project, and run ID |
| Evaluation | Same compatible result directory; always resume mode |
| Dashboard | Same original experiment name and API-returned project ID |
| Slurm | New attempt IDs are appended to the same dashboard project |

Keep failed attempt IDs as history. When moving a run between users, transfer
or expose its checkpoint directory, `wandb_run_id.txt`, compatible partial
results, and grid receipt, then verify the destination can access the same W&B
run.

## Stop conditions

Stop rather than guessing when a submission response is ambiguous, two jobs
may write the same output, a checkpoint/result manifest is incompatible, the
dashboard returns a different project ID, credentials or artifacts cannot
preserve identity, or lowering an OOM-causing microbatch conflicts with a saved
checkpoint. Never blindly resubmit, fabricate IDs, edit checkpoint metadata,
or use evaluation overwrite for an ordinary retry.
