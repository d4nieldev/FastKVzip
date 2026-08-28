# Evaluation Resume and W&B Decision Audit

This document records every material decision in this PR. User decisions quote
the strongest conversation evidence. Choices made while coding stay labeled
**Implementation decision**, even when they are part of the approved plan.

## Run identity

| Decision | Why | Alternatives | Approval |
|---|---|---|---|
| Store one run under `results/<run-name>/`, with `manifest.json`, `metrics.json`, and `outputs/`. | All artifacts for one evaluation are easy to find and move together. | Keep benchmark-first folders or separate manifest and metrics directories. | User formulation: “I want to just have `results/<run-name>/` dir with `manifest.json`, `metrics.json`, and the model outputs.” |
| Do not write `metrics.txt`; print readable metrics in the Slurm log and store structured metrics in `metrics.json`. | This keeps one durable machine-readable metric file without duplicating it. | Keep both JSON and text files. | User answer: “JSON only.” |
| Keep exactly four manifest fields: resolved checkpoint path, whole-file SHA-256, protected-window size, and pruning level. | These are the strict inputs that can change evaluation meaning without being represented in result files. | Store task coverage, ratios, microbatches, Git state, or W&B metadata too. | User formulation: “keep the manifest as minimal as possible and contain only the things we need in order to make sure that we evaluate with the same settings.” |
| Do not store a Git commit or opaque schema/protocol number. | Unrelated code changes must not block resume, and the current format does not need speculative versioning. | Store and enforce the commit; introduce manifest and protocol versions. | User formulation: “remove the git commit from there”; approved plan removes schema and protocol numbers. |
| Require both the resolved absolute checkpoint path and SHA-256 of every byte in the checkpoint file. | The path detects a moved checkpoint; the digest detects replacement at the same path. | Compare only paths, compare only weight tensors, or allow the same file at a new path. | User formulation: “We should also put the path to the checkpoint in the manifest” and “checkpoint path can change while resuming but I disagree.” |
| Keep pruning level fixed for a run. | `pair` uses one global budget; `pair-head` gives every layer/head an equal budget; `pair-layer` gives every layer an equal total budget; `adakv-layer` adds the existing per-head safeguard. | Store a level per output or allow mixed policies. | Approved plan: manifest includes `level` and defines every value. |

## Resume and result files

| Decision | Why | Alternatives | Approval |
|---|---|---|---|
| Use `--existing-results {fail,resume,overwrite}`, defaulting to `fail`. | Existing results are protected unless reuse or deletion is explicit. | Resume automatically or expose separate flags. | User answer: “Require resume flag.” |
| Let `resume` create a run when it is absent. | The same command works for first and later parallel submissions. | Fail when the run is absent. | User answer: “Create a new run (Recommended).” |
| Let `overwrite` permanently remove only the exact run directory and refuse while it is active. | The requested reset is precise and cannot race live writers. | Archive old runs or merge them. | User answer: “Delete permanently”; approved plan. |
| Treat concrete task, example index, and requested ratio as the completion key. | A repeated command can skip exactly the expensive work already saved. | Skip only whole examples or only whole tasks. | User formulation: “skip benchmarks at retention ratios/examples I already evaluated.” |
| Keep task, example, ratio, and full-answer coverage out of the manifest. Derive their union from output files. | Output files are the source of truth and do not require synchronized progress metadata. | Maintain mutable coverage lists in the manifest. | User formulation: “why do we need this in the manifest for we can just look at the files.” |
| Add `_meta` with concrete task, index, dataset size, input fingerprint, and QA-format keys to each output. | Resume can validate dataset identity and completeness without changing generated-answer records. | Trust file paths and list lengths only, or keep a separate task manifest. | Approved plan. |
| Save after each complete ratio using an atomic replacement. | An interrupted job preserves earlier ratios and never exposes partial JSON. | Save only after the whole example or write in place. | Approved plan. |
| Reject corrupt, duplicate, inconsistent, or input-mismatched output instead of repairing it. | Silent repair could combine results from different inputs or settings. | Overwrite suspicious data automatically. | Approved plan. |
| Full-cache answers are additive and are not a manifest invariant. | Resuming in either full-cache mode cannot destroy valid work. | Reject mode changes or keep separate runs. | User formulation: “Full cache answer mode should not reject runs.” |
| When enabled during resume, fill missing full answers only for the selected tasks and index range. | The command changes only the coverage the user selected. | Backfill the whole task or whole run. | User answer: “Selected examples only (Recommended).” |
| Preserve and reuse existing full answers even when `--no-full-cache-answer` is passed. | Disabling generation should avoid new work, not delete useful data. | Clear or ignore existing full answers. | User formulation: “we just add full cache dont delete it.” |
| Allow different benchmarks to compute concurrently under one run name. | Task outputs are independent and parallel evaluation saves wall time. | Allow only one writer for the whole run. | User answer: “Yes” to concurrent same-run jobs. |

## Metrics and W&B

| Decision | Why | Alternatives | Approval |
|---|---|---|---|
| Write absolute, relative, and actual-retention metrics per concrete task. | The three curves show quality, quality relative to full cache, and achieved compression. | Log only absolute performance or only a cross-task aggregate. | User formulation: requested `test/<task>`, `test/<task>-relative`, and `test/<task>-actual-retention`. |
| Keep absolute and relative scores on a 0–100 scale; keep actual retention on a 0–1 scale. | These match the existing score display and the natural retained fraction. | Store every value as 0–1 or every value as a percentage. | Approved plan. |
| Use requested retention as the common x-axis. | Requested ratios are stable across examples; actual ratios vary because protected tokens remain. | Group results by actual retention. | User formulation: actual retention should use requested retention on x and actual retention on y. |
| Average actual retention equally across examples and keep it on a 0–1 scale. | This matches the benchmark's equal-example aggregation and makes it directly comparable with the requested ratio. | Weight by token count or report only per-example values. | User answer: “Mean per example (Recommended).” |
| Compute relative performance as task score divided by that task's full-cache score. | It matches the repository parser's task-relative calculation. | Normalize by another run or average per-example ratios first. | User formulation: requested “average relative performance on the task”; approved plan. |
| Omit relative and full-cache points when the baseline is incomplete or zero. | The ratio is undefined; absolute retained-cache results remain useful. | Fail all metric generation or invent a zero value. | User answer: “Omit relative curve (Recommended).” |
| Keep the cross-task aggregate in `metrics.json` but do not upload it. | It preserves existing parser information while W&B shows the requested task panels only. | Remove the aggregate or add another W&B plot. | Approved plan. |
| Permit W&B upload only when every stored task/ratio covers the full repo-loaded benchmark. | Every uploaded point must represent a final benchmark, never a pilot. | Upload partial aggregates with coverage labels. | User formulation: “if a result is logged in weights and biases then it is the result of the full benchmark.” |
| Treat current full SQuAD coverage as 101 contexts. | The repository loader stops only after adding the 101st distinct context. | Treat the helper's old `--num 100` default as complete. | Approved plan: “With the current loader, full SQuAD coverage means 101 contexts.” |
| Add an `x=1.0` point only for a complete, nonzero full-cache baseline. | It makes full-cache performance visible without publishing a partial or undefined relative baseline. | Never add full-cache points or add partial points. | Approved plan. |
| Rebuild `metrics.json` atomically from every valid output file. | The file always describes the current on-disk union after resume or parallel tasks. | Incrementally patch aggregate metrics during evaluation. | Approved plan. |
| Store example count, dataset size, and completeness for every task and ratio. | Partial local results remain useful and clearly labeled. | Store scores only or reject all partial parsing. | Approved plan. |
| Report mixed full-answer coverage in `metrics.json`. | Full answers are additive and may be complete for only part of a task. | Reject mixed coverage or hide partial full-cache scores. | Approved plan. |
| Use local outputs to skip evaluation; use W&B history only to deduplicate metric upload. | An aggregate W&B point cannot prove which local examples exist. | Trust remote history as evaluation progress. | User clarification: compare W&B against local files; approved plan. |
| Skip matching remote metric points, add missing points, and fail on changed or duplicate points before mutation. | Retries are idempotent and cannot silently rewrite history. | Append duplicates or create revision keys. | User formulation: “if the local metric value and w&b metric value matches (then if not we fail).” |
| Identify the training W&B run from `wandb_run_id` in the checkpoint. | Evaluation metrics attach to the run that produced those exact checkpoint bytes. | Ask for a run ID separately or create an evaluation run. | Approved plan: compare against “the exact training W&B run.” |
| Require `--log-to-wandb` plus an explicit project; keep entity optional. | Upload is deliberate and identifies the training-run namespace. | Log automatically or infer the project. | User answer: separate boolean and required project. |
| Perform no W&B lookup or mutation without `--log-to-wandb`. | Local parsing stays offline and cannot alter a training run by accident. | Always compare remote history after parsing. | Approved plan and verification list. |
| Preserve local files but fail the job when W&B upload fails. | Expensive evaluation work remains recoverable while automation sees the failure. | Report success or delete results. | User answer: “Keep files, fail job.” |
| Allow metrics and W&B post-processing to run without loading the LLM. | An upload retry should not repeat expensive evaluation. | Make upload available only at the end of evaluation. | Approved plan. |

## Delivery

| Decision | Why | Alternatives | Approval |
|---|---|---|---|
| Implement with Ponytail full in an isolated worktree and branch. | The change stays reviewable and avoids unnecessary dependencies or abstractions. | Work directly on `main` or use Superpowers. | User formulation: “work in an isolated worktree and branch” and “use the ponytail skill.” |
| Open a PR to `main`, but do not merge it or submit cluster experiments. | The user wants to review and run jobs personally. | Merge or submit jobs automatically. | Approved plan. |
| Leave `eval.py`, training, checkpoint formats, requirements, and legacy result files unchanged. | This PR is limited to graph-evaluation orchestration and reporting. | Migrate all evaluation paths or add dependencies. | Approved plan. |

## Implementation decisions

This section is updated while coding. Each entry remains explicitly owned by
the implementation rather than being retroactively attributed to the user.

### Run store

| Decision | Why | Alternatives | Approval |
|---|---|---|---|
| Use standard-library POSIX `flock`; create no lock files and add no dependency. | Directory and manifest file descriptors already exist and can coordinate processes. | Add `.lock` artifacts or a locking package. | **Implementation decision.** |
| Hold shared run-directory and manifest locks for an evaluator's lifetime. | Overwrite detects active users, and final metrics wait for all evaluators to finish. | Trust only a brief manifest lock or use a database. | **Implementation decision.** |
| Lock the results root for run creation, each task directory for merges, and the manifest exclusively for finalization. | Only short metadata/write sections serialize; different benchmark computation stays parallel. | Lock the whole run during computation or use one lock per ratio. | **Implementation decision.** |
| Accept only one safe path component for run and task names. | Overwrite and merge targets remain exact. | Allow nested paths with containment checks. | **Implementation decision.** |
| Hash checkpoints in 1 MiB chunks. | The SHA covers every byte without loading a large checkpoint into RAM. | Read the whole file before hashing. | **Implementation decision.** |
| Canonicalize only `context`, `question`, and `answers` as compact, sorted UTF-8 JSON for the input SHA. | These are the approved evaluation inputs; unrelated loader fields do not block resume. | Hash the whole dataset record or raw Python representation. | **Implementation decision.** |
| Require exact manifest and output metadata keys, then strictly validate every stored answer entry. | Corrupt or mixed results fail before they are trusted. | Ignore unknown fields or parse best-effort. | **Implementation decision.** |
| Require canonical output filenames such as `0.json`; reject aliases such as `00.json`. | Two paths cannot represent and double-count the same example index. | Track duplicate numeric indices while parsing. | **Implementation decision.** |
| Require formats to share ratio order, actual retention, and threshold; keep each format's ground truth stable across ratios; require all-or-none full answers. | One ratio merge represents one coherent evaluation of the example. | Permit inconsistent ratio metadata or partial QA-format coverage. | **Implementation decision.** |
| Reject an identical repeated ratio as a duplicate and a changed repeated ratio as a conflict. | A race cannot silently create or replace a completion key. | Accept identical retries or use last-writer-wins. | **Implementation decision.** |
| Propagate each QA format's full answer into every stored ratio entry for that format. | Full-cache output is format-level data and remains consistent after later backfill. | Store it only in the ratio that generated it. | **Implementation decision.** |
| Write JSON beside its target, flush and `fsync`, then replace the target atomically. | Readers see the old file or the complete new file, never a partial write. | Write in place or keep journal files. | **Implementation decision.** |
| Re-hash the checkpoint before standalone metric/W&B post-processing. | A missing or replaced checkpoint cannot receive metrics from stale local outputs. | Trust the manifest after evaluation. | **Implementation decision.** |
| Expose validated examples in lexical task and numeric index order. | Metric output and tests stay deterministic. | Return raw directory iteration order. | **Implementation decision.** |

Local multiprocessing tests cover these locks. Cross-node behavior still
depends on the cluster's shared filesystem supporting POSIX `flock`; verify
that once before launching concurrent benchmark jobs.

### Evaluation integration

| Decision | Why | Alternatives | Approval |
|---|---|---|---|
| Preserve the legacy `eval_graph.py` save path when `--run-dir` is omitted. | Existing direct commands and upstream-style tests keep working; the helper always selects the new run store. | Make the new run directory mandatory or change shared `save_result`. | **Implementation decision.** |
| Reject duplicate requested ratios before model construction. | Duplicate work would otherwise create ambiguous completion keys. | Deduplicate silently or let output validation fail later. | **Implementation decision.** |
| Validate and open the run before constructing the GPU runtime. | Manifest conflicts fail before expensive LLM allocation. | Build the model first as the old evaluator did. | **Implementation decision.** |
| Check local completion before prefill. | A completed task/example/ratio avoids the expensive LLM and mixer work. | Prefill first and skip only generation. | **Implementation decision.** |
| Prefill without hidden-state capture and skip mixer scoring when only a full answer is missing. | Full-cache generation needs the KV cache, not pruning scores. | Capture hidden states and run the mixer anyway. | **Implementation decision.** |
| Reuse a stored full answer when adding ratios. | Resume generates only the missing retained-cache outputs. | Regenerate full answers for every new ratio. | **Implementation decision.** |
| Compare generated QA-format keys with the dataset's question count before every merge. | A first write cannot silently omit a question and later appear complete. | Trust the evaluator's returned keys. | **Implementation decision.** |
| Merge each ratio only after every QA format finishes. | A saved completion key always represents a complete ratio. | Save each QA format independently or wait for the whole example. | **Implementation decision.** |
| Show a completed cached progress entry without tokenizing or prefilling it. | Obtaining an exact token count would defeat part of the skip. | Retokenize cached examples for the progress postfix. | **Implementation decision.** |

### Metrics and W&B implementation

| Decision | Why | Alternatives | Approval |
|---|---|---|---|
| Reuse `EvaluationRun.iter_examples()` as the parser's only output validator. | One strict schema implementation avoids drift. | Duplicate validation in the parser. | **Implementation decision.** |
| Import dataset helpers, Torch, and W&B only in code paths that need them. | Local metrics and legacy commands do not load unrelated heavy packages. | Import every dependency at process start. | **Implementation decision.** |
| Store unrounded numeric values in `metrics.json`; round only readable terminal output. | Machine-readable data keeps full precision. | Round the JSON to display precision. | **Implementation decision.** |
| Compute relative performance from aggregate task score divided by aggregate full-cache score. | This matches the repository's legacy parser. | Average per-example relative ratios. | **Implementation decision.** |
| Query W&B with stable `scan_history()` for all three expected keys of every local task. | It returns unsampled rows, finds partial uploads, and detects duplicate remote points even when a local relative curve is omitted. | Fetch all training history at once or use the beta history API. | **Implementation decision.** |
| Resolve an omitted entity through `wandb.Api().default_entity`. | The public lookup and resumed writer use one exact entity. | Require entity on every command. | **Implementation decision.** |
| Require the target W&B training run to be finished. | Evaluation should not mutate a still-running training history. | Permit upload to running or failed runs. | **Implementation decision.** |
| Validate all remote points before calling `wandb.init`. | Conflicts and duplicates fail before W&B mutation. | Validate and upload one point at a time. | **Implementation decision.** |
| Compare local and remote values with `1e-9` relative and absolute tolerance. | W&B serialization may introduce harmless floating-point noise, while material changes still fail. | Require bitwise equality or use a looser display-level tolerance. | **Implementation decision.** |
| Ignore and preserve remote task/ratio points outside the current local union. | Separate post-processing scopes can add new ratios without requiring unrelated local files. | Reject every remote point absent from the current run directory. | **Implementation decision.** |
| Group only missing metrics by requested ratio into minimal W&B rows. | Partial retries add only absent absolute, relative, or actual-retention values. | Re-log complete rows or one row per metric. | **Implementation decision.** |
| Define every task metric explicitly against `test/retention_ratio`. | W&B builds the requested per-task curves without wildcard rules. | Use implicit step axes or a wildcard definition. | **Implementation decision.** |
| Disable W&B system statistics and metadata for the post-processing writer. | Upload adds only the requested test curves and no second set of machine panels. | Use the default SDK monitors. | **Implementation decision.** |
| Finish the resumed W&B writer with exit code zero even when upload raises. | The evaluation job fails, but the already-finished training run is not relabeled as failed. | Mark the training run failed because post-processing failed. | **Implementation decision.** |
| Hold exclusive finalization across metric rebuild, atomic write, remote comparison, and upload. | W&B and `metrics.json` describe one stable union of output files. | Release the lock before network upload and recheck later. | **Implementation decision.** |
| Make the parser, not the shell, atomically write `metrics.json`. | One process owns metric structure and crash-safe replacement. | Pipe parser text through `tee` or assemble JSON in Bash. | **Implementation decision.** |

### Shell integration

| Decision | Why | Alternatives | Approval |
|---|---|---|---|
| Let the submission helper own `--run-dir`, `--tag`, and result-mode handling. | One validated run name determines the Slurm name and every artifact path. | Ask users to repeat paths and tags manually. | **Implementation decision.** |
| Pass batch-only controls before `--` and forward only arguments after it to `eval_graph.py`. | W&B and lifecycle flags cannot accidentally reach the model CLI. | Encode controls in environment variables or duplicate all evaluator flags in shell. | **Implementation decision.** |
| Keep the Git commit in the Slurm log, but not in the manifest or resume validation. | It is useful provenance and cannot reject a run. | Remove it completely or make it an invariant. | **Implementation decision.** |
| Let the Python evaluator own overwrite after the Slurm job starts. | Active-run locking and exact deletion live in one place. | Delete results in the helper before submission. | **Implementation decision.** |
| Run metrics only after successful evaluation in the same batch job. | Shell error handling skips parsing after evaluator failure and fails on parser/upload errors. | Submit a separate dependent job. | **Implementation decision.** |
| Let the parser put W&B SDK files in a temporary directory and remove it when upload ends. | Batch and direct retries cannot create a fourth durable artifact location or dirty the repository. | Manage `WANDB_DIR` in each caller or keep `wandb/` beside results. | **Implementation decision.** |

## Files in this PR

Added:

- `prefill/results/evaluation_run.py`
- `prefill/tests/test_evaluation_run.py`
- `prefill/tests/test_result_parse_run.py`
- `docs/evaluation-wandb-decisions.md`

Modified:

- `prefill/eval_graph.py`
- `prefill/results/parse.py`
- `prefill/tests/test_graph_eval.py`
- `slurm/eval_graph.sbatch`
- `slurm/submit_eval_graph.sh`
- `docs/graph-fastkvzip-experiments.md`
- `docs/graph-fastkvzip-decision-audit.md`

`test_result_parse_run.py` keeps parser and W&B tests independent of Torch.
The old decision audit gets only a superseded link so it does not contradict
the new run layout. Both are **Implementation decisions.**
