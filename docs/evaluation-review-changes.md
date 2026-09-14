# Evaluation PR Review Changes

Do not implement it until the review ends.

## 1. Remove locking

**Problem.** The code protects same-task writers and other races we do not support.

**Implement.**

- Remove `fcntl` and all run, manifest, task, and finalization locks.
- Parallel jobs may share a run only when they evaluate different tasks.
- Jobs sharing a run must use the same manifest values.
- Ignore unsupported overlap and concurrent `metrics.json` replacement.

**Decision audit.** [Resume behavior](evaluation-wandb-decisions.md#resume-behavior): “Allow jobs for different tasks to use the same run name.” [Under review](evaluation-wandb-decisions.md#under-review): `File locking`.

## 2. Remove checkpoint hashing

**Problem.** Hashing the full checkpoint adds code and repeated file reads.

**Implement.**

- Remove `checkpoint_sha256` from the manifest and code.
- Store the resolved checkpoint path and training W&B run ID.
- Compare those values when resuming.
- Do not detect changed bytes at the same path.

**Decision audit.** [Run identity and storage](evaluation-wandb-decisions.md#run-identity-and-storage): “Keep the manifest small.” [Under review](evaluation-wandb-decisions.md#under-review): `Checkpoint SHA-256`.

## 3. Remove `_meta` and input hashing

**Problem.** `_meta` duplicates information available elsewhere.

**Implement.**

- Infer the task from the parent directory.
- Infer the example index from the filename.
- Infer question keys from the JSON keys.
- Load the task dataset to obtain its size.
- Remove `input_sha256`.
- Keep the original FastKVzip output shape.

**Decision audit.** [Resume behavior](evaluation-wandb-decisions.md#resume-behavior): “Read task, example, ratio, and full-answer coverage from output files.” [Under review](evaluation-wandb-decisions.md#under-review): `Input SHA-256` and `Stored JSON validation`.

## 4. Deduplicate retention ratios

**Problem.** Repeated requested ratios are treated as errors in several places.

**Implement.**

- Deduplicate command-line ratios during argument parsing.
- Skip ratios already stored for the example.

**Decision audit.** [Resume behavior](evaluation-wandb-decisions.md#resume-behavior): “Track completed work by task, example index, and requested retention ratio.” [Under review](evaluation-wandb-decisions.md#under-review): `Repeated requested ratios`.

## 5. Remove Git commit logging

**Problem.** The Slurm log does not need the repository commit.

**Implement.** Remove the Git commit log line from `eval_graph.sbatch`.

**Decision audit.** [Shell behavior](evaluation-wandb-decisions.md#shell-behavior): “Print the Git commit in the Slurm log only.”

## 6. Finalize metrics after each benchmark

**Problem.** Metric parsing currently waits for the whole evaluation command to finish.

**Implement.**

- After one concrete benchmark finishes, calculate its metrics immediately.
- Rebuild `metrics.json` from the available outputs.
- Upload that benchmark's completed points to W&B immediately.
- Do not wait for other benchmarks selected by the same command.
- Do not upload an incomplete benchmark.

**Decision audit.** [Metrics and W&B](evaluation-wandb-decisions.md#metrics-and-wb): “Upload to W&B only when every stored task and ratio is complete.” [Shell behavior](evaluation-wandb-decisions.md#shell-behavior): “Run metric parsing only after evaluation succeeds.”
