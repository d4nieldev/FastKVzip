# RULER Integration and Prioritized Full Evaluation

Approved implementation plan. Implementation is isolated in
`.worktrees/ruler-evaluation`, branch `feature/ruler-evaluation`, based on freshly
fetched `origin/main`. Preserve the original dirty workspace.

Production was approved and all 107 jobs submitted on **2026-09-08**. The
[production manifest and receipt](../ruler-production-manifest.md) records every
job ID. At the first check, 15 were running, 92 were queued, and **0/107
benchmarks were complete**; submission is not completion.

## 1. Dataset and evaluator changes

Support RULER's 13 tasks across 4K, 8K, 16K, 32K, 64K, and 128K using pinned
`lighteval/RULER-{length}-Qwen2.5-Instruct` datasets and standard Hugging Face caching.
Keep selection inside `--data`: `ruler`, `ruler_128k`,
`ruler_niah_single_1_128k`, and `all`. No preparation command, data-generation
component, data-directory flag, or separate evaluation framework.

- Normalize published rows into context/question/answers, preserving demonstrations
  and multiple references. Keep the final question outside prefill and compression
  scoring.
- Share complete, deduplicated dataset expansion between evaluation and result
  parsing. Include supported SCBench length variants without automatic substitutions.
- Exclude Agentic from `all` and all evaluation jobs; preserve its existing
  loader/training functionality.
- Evaluate complete fixed benchmarks by default, with `--num` as an optional limit.
  Correct loader-limit plumbing and SQuAD context-count handling without losing
  questions.
- Reuse existing prefill, graph scoring, subgraphs, pruning, generation, and result
  storage.
- Retain `--ruler-prompt-mode {graphkv,official}`, default `graphkv`. This changes
  task/template wrapping only; distinguish modes in result identities.
- Add official scoring through existing metric dispatch: NIAH/VT/CWE/FWE use
  case-insensitive substring recall across targets; QA matches any accepted alias.
  Output limits: NIAH 128, VT 30, CWE 120, FWE 50, QA 32 tokens.
- Shared generation removes the final token only if it matches an effective
  configured EOS ID. Preserve capped non-EOS output, handle empty output and multiple
  EOS IDs, and prevent incompatible old caches/results from silently mixing with
  corrected generation.

## 2. Verification and pilots

Develop focused tests first, then run the complete existing suite and compilation
checks. Cover conversion, all selectors/counts, prompt modes, metrics, EOS handling,
resume compatibility, W&B destination binding, and production manifest construction.
Commit, push, and open the PR to `main` before pilots.

Pilot the uniform checkpoint on one example from every RULER task at 4K in both
prompt modes, and one example from `niah_multikey_3`, `vt`, and `qa_2` at 128K in
both modes. Use production settings:

```text
--level pair
--window-size 0
--ratios 0.75 0.50 0.40 0.30 0.20
--full-cache-answer
--existing-results resume
```

Pilots make no W&B writes. Manually inspect questions, references, outputs, scores,
and retention behavior. Record runtime/memory measurements for production sizing.
Old window-0.02 compressed scores are a different configuration, not an exact
regression target for window 0.

## 3. Checkpoint staging and production submission

Evaluate `q25a-s40n40-n200e2-uniform-s0/best.pt`. Before any production submission:

- Stage it in durable storage accessible to each of `danieloh`, `guyzagor`, and
  `odedshah`. Verify all three copies share SHA-256 and expected model/configuration.
  Record each account's resolved path/checksum.
- Verify matching code commits, compatible environments, storage, and live resource
  entitlements. Stop the entire submission if any account lacks its verified copy.
  Never modify checkpoint metadata to change logging identity.

Submit 107 independent benchmark jobs, each covering its complete prepared/filtered
split, all five retention ratios, and the full-cache baseline:

| Submission order | Benchmarks | Jobs |
| --- | --- | ---: |
| 1 | `scbench_kv` | 1 |
| 2 | All RULER 4K tasks | 13 |
| 3 | All RULER 8K tasks | 13 |
| 4 | Remaining RULER, SCBench, SQuAD, and GSM configurations | 80 |
| | Total | 107 |

Production uses GraphKV prompts. These are submission-order tiers, not completion
dependencies: capture all earlier-tier IDs before submitting the next tier, without
waiting for completion. Slurm may still start jobs out of submission order.

Use the standard evaluation wrapper with one GPU/job and justified workload-specific
requests. Within tiers, balance estimated GPU-hours across available account slots
using pilot measurements, dataset size, query count, and context length; distribute
expensive jobs, not merely job counts. Target maximum permitted concurrency across
three accounts (up to 15 if five each), respecting shared limits/unrelated work.
Present the exact manifest, assignments, and resource requests for approval before
submission. Preserve required scheduler prerequisites; stop if unresolved.

## 4. One new evaluation-only W&B run

After pilots pass, create exactly one new run:

- Project: `graphkv-answer-qwen25-7b1m-s40n40-grid-v1`
- Name: `q25a-s40n40-n200e2-uniform-s0-eval-all-w0`
- Job type: `evaluation`

Record source training run, checkpoint checksum, dataset revisions, prompt mode,
and evaluation settings in configuration. Do not write to the old training run.

Add optional `--wandb-run-id` binding: explicit ID overrides checkpoint destination
without changing the checkpoint; persist resolved destination in the existing
evaluation manifest; allow binding without immediate logging; honor it in both
evaluators sharing the parser; preserve omitted-flag behavior.

All production GPU workers bind the new ID but leave direct uploads disabled. One
serial coordinator promptly uploads each completed benchmark, reusing completeness
checks, conflict detection, and resumable uploads. Priority results appear without
waiting for all jobs. Preserve normal `test/<task>` metric names and the retention
axis. Retries reuse this new run ID.

## 5. Completion and reporting

Use isolated durable result directories, resume compatible partial work, and allow
one writer per directory. Preserve every job attempt in the grid receipt and
existing dashboard project. Tests specifically enforce:

- 107-job inventory and submission order; three-account checkpoint gate.
- No Agentic jobs or pilot W&B writes.
- One shared production destination, no training-run writes.
- Correct resume behavior and idempotent incremental uploads.

Monitor through completion, investigate failures, and resume safely. Report
per-task scores, retention diagnostics, RULER macro-averages per length/ratio, and
explicit coverage. Update the experiment-results document and PR with pilot
evidence, production job IDs, completed results, and unresolved failures. Delivery
includes the verified PR and production results, not merely successful submission.

## Verification and execution receipt (2026-09-08)

- Worktree base: `d7e18d7f989036a84ab26eac59a586a0d65fb87f` (`origin/main`
  freshly fetched before creating the branch). Original dirty workspace preserved.
- Baseline prefill tests: 219 passed. Current implementation: 405 passed.
- Compilation (`compileall` over `prefill` and `slurm`), standard evaluation
  wrapper shell syntax, and `git diff --check` passed.
- Actual pinned Hugging Face loader smoke: one row from each of the thirteen
  4K splits loaded successfully. This is not a GPU pilot.
- The full bundled LaTeX suite was also run: 356 passed, 429 failed. A clean
  archive of the base commit gives the same counts and identical failing test
  IDs. No files under `math/` changed. These unrelated baseline failures remain.
- BGU SSH authentication succeeded for all three approved accounts. Accounting
  confirms source training `21061828` and original evaluation `21061591` completed
  with exit `0:0`; the live scheduler no longer resolves `21061828` for `afterok`.
  At this pre-pilot check, no evaluation jobs had been submitted and no W&B
  runs/results had been created.
- On 2026-09-08 the user explicitly approved omitting the expired
  `afterok:21061828` prerequisite for this evaluation. Keep that completed job as
  provenance. The planner records this narrow exception as
  `completed_checkpoint_approval` with matching job ID, approval date and archived
  approval reference; without that record the prerequisite is still required.
  Three-account checkpoint staging/verification and approval of the measured
  production resource manifest remain mandatory before production.
- First pilot batch: nine successful scheduler exits, including the approved
  RULER coverage and a three-context SQuAD timing check. All three staged copies,
  saved prefixes, model revision and runtime package versions matched. See the
  experiment ledger for exact paths and job IDs.
- The short-context pilots exposed an existing window-resolution bug: explicit
  integer zero became a protected 2% tail. Fixed zero explicitly and identified the
  corrected behavior with evaluation-only `window_revision=1`; older manifests
  remain readable as revision 0 but cannot resume or enter production uploads.
  This does not change full-context teacher-answer cache identity. Preserved and
  reran the two 4K pilots and SQuAD timing check; retained unaffected 128K evidence.
- Corrected pilot jobs `21113328`, `21113329`, `21113330` passed at runtime
  commit `7323c0d`; all three account checkouts are clean at that commit.
  [Pilot evidence](../ruler-pilot-evidence.md) records exact sampled scores and
  measurements, including the unaffected long-context attempts.
- Created exactly one evaluation-only W&B run, `hdo2z4x4`, after the pilot gate
  passed. No pilot metrics were uploaded; no production results were ready to
  upload at the first status check (2026-09-08 06:47:59 UTC).
- The user approved the [107-job manifest](../ruler-production-manifest.md) and
  dashboard job-ID export on 2026-09-08. All 107 production jobs were submitted
  under the unchanged approval digest
  `30adbdab4df476c1c05c82a75f911b4172e4c84fd58abac771ee372fd0842818`;
  exact IDs, commands, and submission tiers are preserved in `receipt.json`.
  Live effective partition `gpu` / partition QoS `gpu-part` confirms five GPUs
  per user, and test-only scheduler validation passed on all three accounts.
  The submitted grid totals about 492 estimated GPU-hours, not measured full-run
  runtime or a promised completion time.
- First status check: 15 jobs running (five per account), comprising SCBench KV,
  all 13 RULER 4K tasks, and `ruler_niah_multiquery_8k`. The other 92 were pending
  with `QOSMaxGRESPerUser`, not completion dependencies. Benchmark completion was
  **0/107**; monitoring and serial completed-result uploads remain in progress.
- After explicit export permission, dashboard registration succeeded for all
  12 pilot attempts and 107 production attempts in the existing
  `graphkv-answer-qwen25-7b1m-s40n40-grid-v1` project. No checkpoint contents,
  generated outputs, or credentials were sent to that dashboard.
