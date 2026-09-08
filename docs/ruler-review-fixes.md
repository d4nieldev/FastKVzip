# RULER review fixes before redistribution

Verified 2026-09-08 in the isolated `feature/ruler-evaluation` worktree, addressing
the four required findings in [PR #22's logic review](https://github.com/d4nieldev/FastKVzip/pull/22#issuecomment-5589487960).

- MRCR accepts an unlimited `--num` and reports its filtered cardinality only
  after exhausting the source. Bounded reads remain bounded and report an
  unknown total.
- Dataset selection no longer accepts the unused model argument. All callers,
  including the legacy chunked evaluator, use explicit dataset selectors;
  automatic SCBench substitutions stay removed.
- Unknown dataset sizes are persisted as `null`, never inferred from a slice.
  Local metrics are available, but these slices cannot be uploaded as complete
  benchmarks. A subsequent full read can establish the previously unknown size.
- RULER downloads just the requested pinned task's Parquet file through the
  native Hugging Face Hub cache, then reads it through Datasets. It does not
  prepare the other twelve tasks. Prompts, row conversion, revisions and scoring
  are unchanged.

The new tracked `slurm/collect_ruler_evaluation.py` retains the existing serial
collector state and immutable original grid receipt. An explicitly approved
retry receipt records new attempts and their account-specific locations. Only
zero-output jobs may move accounts; partial results stay in their original
directory. Collection and uploads exclude tasks with possible active writers,
including retries that appear during a collection pass. Upload interruptions
remain resumable, with the same evaluation-only W&B destination and all failed
attempts retained.

## Verification

- Complete `prefill/tests` suite: **426 passed** (Python 3.12.1, CPU).
- `python -m compileall -q prefill slurm`, shell syntax check of
  `slurm/eval_graph.sbatch`, and `git diff --check`: passed.
- Native Hub/Datasets cache regression: a second process with HTTP forbidden
  reads all 500 synthetic Parquet rows and requested ranges from the warm cache.
- Live public-data comparison: all 500 normalized `ruler_qa_1_4k` rows are exactly
  equal between the previous streaming path and the new cached path. Canonical
  compact sorted-JSON SHA-256:
  `0b08419845bf1430f7c303e280a37512af16083e879c12298f021ac4aaa1db2a`.
- Live metadata checks: all six pinned repositories have the expected thirteen
  one-shard task files (78 total).
- Read-only collector dry run accepts the actual original 107-attempt receipt
  with SHA-256
  `30adbdab4df476c1c05c82a75f911b4172e4c84fd58abac771ee372fd0842818`.
- Additional collector regressions cover account routing, partial-output guards,
  duplicate/unknown attempts, changing receipts, interrupted uploads and keeping
  previously collected snapshots out of uploads while a retry is active.

No deployed checkout, production job, result or W&B run was changed during this
fix-and-verification pass. Existing jobs may continue on their unchanged code.
These fixes do not require rerunning already reported production scores. Retry
placement, runtime staging and exact scheduler requests still require approval;
the completed RTX 6000 setup smokes covered only 4K, not long-context capacity.
