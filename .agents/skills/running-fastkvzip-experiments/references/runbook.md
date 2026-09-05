# Grid Submission Runbook

## 1. Discover eligible users

From the skill directory, run:

```bash
python3 scripts/discover_slurm_users.py --anchor bgu-slurm
```

The helper reads SSH config/includes and uses `ssh -G`; it does not connect.
Pass repeated `--alias NAME` arguments for explicitly supplied aliases. Treat
its output as candidates, not authorization.

For every candidate, apply the bounded VPN/SSH gate and record:

| Field | Required evidence |
|---|---|
| SSH alias / Unix user | `ssh -G` plus bounded authentication |
| Slurm eligibility | live partition, account/QoS, typed GPU, and limits |
| Project | exact clean commit and remote path |
| Runtime | venv, model revision/access, dependency check |
| Storage | checkpoint, result, cache capacity and sharing |
| Cache | full activations, scores-only, or isolated/no-save behavior |
| W&B | accessible entity/project and persistent run IDs |
| Dashboard | account's jobs are ingested by its dashboard agent |

Do not use additional identities to evade site quotas. Exclude an identity that
cannot preserve the required artifacts or W&B identity.

## 2. Materialize and assign the grid

Expand the Cartesian product before submission. Give each row a deterministic,
unique logical run name. Record common arguments once and every varying field
explicitly; do not rely on remembered defaults.

Assign the whole training/evaluation pair to one eligible identity. Respect
cache/storage constraints first. Then sort rows by expected GPU-hours descending
and logical run name ascending, then place each on the currently least-loaded
identity; break worker ties by SSH alias. If durations are unknown, use equal
weights, yielding deterministic round-robin.

Store a durable receipt under an ignored path such as:

```text
.slurm/grids/<experiment-name>/
├── manifest.json
└── receipt.json
```

The manifest contains configurations and assignments. The receipt contains the
original experiment name, W&B entity/project, dashboard project ID, commit,
and, per logical row, every training/evaluation attempt ID and state.

## 3. Preflight

Before showing the submission preview:

- Verify every account uses the intended commit and compatible environment.
- Inspect `sres` and current Slurm limits; justify time and host RAM from
  comparable jobs or pilots.
- Check `squeue` and time-bounded `sacct` for each logical run name.
- Check checkpoint/result paths and persistent W&B IDs.
- Validate the exact FineWeb slice and expected cache keys.
- Ensure one writer per checkpoint directory, result directory, and missing
  shared-cache key.
- Verify evaluation arguments and future checkpoint paths.

A shared cache must be complete before concurrent consumers launch. Otherwise,
submit exactly one cache-producing run and gate consumers on it, or use isolated
caches. Concurrent misses can race even when writes are atomic.

For a full activation cache, pass `--teacher-cache-dir <path>` and omit the
scores-only flag. For storage-constrained operation, pass both
`--teacher-cache-dir <path>` and `--teacher-cache-scores-only`; this keeps the
teacher model resident and therefore needs a pilot for the exact GPU/microbatch
shape. Use a separate cache directory for the two formats. Confirm the current
checkout supports the flag before including it.

Show the exact grid, assignments, resource requests, train commands, evaluation
commands, W&B destination, and dashboard experiment name. Submit only after
explicit approval.

## 4. Submit one train/evaluation pair

For each row, submit training once and parse the numeric portion of the
`sbatch --parsable` response (`JOB_ID` or `JOB_ID;CLUSTER`). Persist the training
ID before the next scheduler mutation.

Immediately submit evaluation with:

```text
--dependency=afterok:<training-job-id>
--graph-checkpoint <future-run-directory>/last.pt
--existing-results resume
```

Use `last.pt` when training uses `--no-save-best`; otherwise use the
user-approved checkpoint policy. Always include `--existing-results resume`,
even for a new result directory.

`slurm/submit_eval_graph.sh` checks that the checkpoint already exists and does
not own scheduler dependencies. To queue evaluation immediately, call
`sbatch --parsable --dependency=afterok:<id> ... slurm/eval_graph.sbatch`
directly with the helper's equivalent environment, resource, name, log, and
result arguments. Do not pass `--dependency` through the helper.

Persist the evaluation ID and its training dependency. One row must have one
active training writer and at most one evaluation capable of writing its result
directory.

### Unknown submission outcome

A timeout or lost stdout after `sbatch` is an unknown commit. Stop all new
submissions. Reconcile the intended `(Unix user, run name)` with `squeue`, a
time-bounded `sacct`, and `scontrol show job -dd`; compare submission time,
working directory, command, resources, and dependency.

- Exactly one match: adopt its ID.
- No match after scheduler/accounting settle and one bounded recheck: submit
  once.
- Multiple or ambiguous matches: stop and present exact IDs. Cancellation needs
  explicit approval.

Never infer IDs from numeric adjacency or retry the whole grid blindly.

## 5. Register the dashboard

The dashboard base URL is `https://graphfastkvzip.onrender.com`. No auth header
is needed.

After collecting every training and evaluation ID in the batch, always create
the project—even during a retry—and never pre-check:

```http
POST /api/projects
Content-Type: application/json

{"name":"<original experiment name>"}
```

Read `id` from the response; never construct it. On a retry/resume it must match
the ID in the receipt, because the exact original name returns the existing
project.

File all IDs from this submission batch in one call:

```http
POST /api/projects/<returned-id>/jobs
Content-Type: application/json

{"job_ids":["<train-id>","<eval-id>","..."]}
```

Verify `assigned` equals the number sent. If not, reconcile dashboard ingestion
and repeat the idempotent dashboard assignment with the same IDs. Never resubmit
Slurm jobs to fix dashboard registration. Keep earlier failed attempt IDs in the
same project and append replacement IDs.

## 6. Monitor and finish

Inspect exact IDs in this order: `squeue`, `scontrol`, the discovered logs,
`sstat` while running or `sacct` when terminal. A pending job is not a reason to
resubmit. An evaluation blocked by `Dependency` is expected while training runs.

A logical row is complete only when:

- final training is `COMPLETED` with `ExitCode=0:0`;
- `last.pt` (or the approved final checkpoint) is valid at the target cursor;
- evaluation is `COMPLETED` with `ExitCode=0:0`;
- expected task/example/ratio and full-cache coverage is complete;
- W&B metrics belong to the original run without conflicts; and
- every initial/replacement Slurm ID is recorded in the same dashboard project.
