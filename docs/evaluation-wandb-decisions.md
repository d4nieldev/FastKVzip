# Evaluation Resume and W&B Decision Audit

This document explains the current PR in pipeline order.

- **User decision** means the user chose it directly or approved it in the plan.
- **Implementation decision** means it was chosen while writing the code.
- **Under review** means the code does it today, but the user has questioned it.

The under-review section describes the current code. It does not argue that the
code should stay that way.

## Pipeline at a glance

1. The submit helper receives a run name, checkpoint, tasks, and retention ratios.
2. The evaluator creates or opens `results/<run-name>/`.
3. For each task and example, it reads `outputs/<task>/<index>.json` if it exists.
4. It skips retention ratios and full-cache answers that are already saved.
5. It runs prefill, mixer scoring, and generation only for missing work.
6. It saves each completed ratio into the example's JSON file.
7. After each task, the evaluator rebuilds `metrics.json` from all saved outputs.
8. If that task is complete, it can upload its test curves to W&B.

## Terms used in this document

| Term | Meaning |
|---|---|
| Evaluation run | One results directory, such as `results/my-eval/`. |
| Training W&B run | The W&B run created while training the checkpoint. It is not the evaluation results directory. |
| Task | One concrete benchmark, such as `squad` or `scbench_qa_eng`. |
| Example | One context at one dataset index. |
| Requested retention | The cache fraction passed on the command line, such as `0.2`. |
| Actual retention | The kept fraction of scored context-cache positions, averaged across layers and heads. Prefix and query positions are outside this score mask. Protected local-window context positions can make it larger than requested. |
| Question key | One question that uses the example's context. Output files name these keys `qa`, `qa-1`, and so on. |
| Pruned answer | The model answer after cache pruning at one retention ratio. |
| Full-cache answer | The optional model answer without pruning. |
| Ground-truth answer | The correct dataset answer. It is stored as `answer`. |
| Pruning level | How the retention budget is divided across layers and KV heads. |

## One output entry

One context may have several questions. Each question key stores one entry per
requested retention ratio:

```json
{
  "qa": [
    [
      [0.2, 0.24, 0.0058],
      {
        "pruned": "answer produced with the pruned cache",
        "full__": "answer produced with the full cache",
        "answer": "correct dataset answer"
      }
    ]
  ]
}
```

The three numbers are requested retention, actual retention, and pruning
threshold.

The task comes from the parent directory. The example index comes from the
filename. The question keys come from the JSON keys. The evaluator supplies
the loaded dataset size when it calculates task metrics.

## User-approved behavior

### Run identity and storage

| Decision | Why | Alternative | Approval |
|---|---|---|---|
| Store one run under `results/<run-name>/`, with `manifest.json`, `metrics.json`, and `outputs/`. | All files for one evaluation stay together. | Keep benchmark-first directories or separate manifest and metrics directories. | User formulation: “I want to just have `results/<run-name>/` dir with `manifest.json`, `metrics.json`, and the model outputs.” |
| Store metrics only as JSON. Print readable metrics in the Slurm log. | This avoids a duplicate `metrics.txt` file. | Save both JSON and text. | User answer: “JSON only.” |
| Identify a checkpoint by its resolved path and training W&B run ID. Do not hash the file. Rewritten bytes at the same path and run ID are not detected. | This avoids reading the whole checkpoint again. | Hash every checkpoint byte. | User formulation: “we can take the wandb run id instead.” |
| Keep only checkpoint identity, protected-window size, and pruning level in the manifest. | These values must match when a run resumes. | Also store tasks, ratios, microbatches, or Git state. | Approved plan and user review. |
| Do not store a Git commit or schema number in the manifest. | Unrelated code changes must not block resume. | Enforce a commit or format version. | User formulation: “remove the git commit from there.” |
| Keep one pruning level and protected-window size for a run. | Mixing these settings would make results under the same run name incomparable. | Store these settings separately for every result. | Approved plan. |

### Resume behavior

| Decision | Why | Alternative | Approval |
|---|---|---|---|
| Use `--existing-results {fail,resume,overwrite}`. Default to `fail`. | Existing results are changed only when the command says so. | Resume automatically. | User answer: “Require resume flag.” |
| Let `resume` create the run when it does not exist. | The same command works for the first and later jobs. | Require a separate create command. | User answer: “Create a new run (Recommended).” |
| Let `overwrite` delete only `results/<run-name>/`. | The deletion target is narrow and predictable. | Archive or merge the old run. | User answer: “Delete permanently.” |
| Track completed work by task, example index, and requested retention ratio. | Resume can skip the exact expensive work that already finished. | Skip only whole examples or whole tasks. | User formulation: “skip benchmarks at retention ratios/examples I already evaluated.” |
| Read task, example, ratio, and full-answer coverage from output files, not from the manifest. | The output files already show what exists. | Maintain mutable coverage lists in the manifest. | User formulation: “why do we need this in the manifest for we can just look at the files.” |
| Identify an input example by task and index. Do not hash its text. | This keeps outputs small and trusts the selected dataset. | Hash the context, questions, and answers. | User question and approval: “Isn't index enough?” |
| Keep result files in the original FastKVzip shape. | Their directory, filename, and JSON keys already identify the task, example, and questions. | Add a `_meta` object and an input hash to every file. | User review change: “can we get rid of `_meta`?” |
| Use the loaded dataset length when calculating task coverage. | The output file does not need to store dataset size. | Store dataset size in every output file. | User formulation: “I have no problem with loading the data to see its size.” |
| Deduplicate requested ratios before evaluation. | Repeating a CLI ratio must not repeat model work. | Reject the command or run the ratio twice. | User review change. |
| Save after each completed ratio. | A stopped job keeps earlier completed ratios. | Save only after the whole example or task. | Approved plan. The exact file-writing method is under review below. |
| Treat full-cache answers as optional, additive data. | Switching full-cache mode during resume does not reject or delete useful results. | Make full-cache mode fixed for the run. | User formulation: “Full cache answer mode should not reject runs.” |
| Backfill full-cache answers only for the tasks and example range selected by the command. | Resume does not start unrequested work. | Backfill the entire run. | User answer: “Selected examples only (Recommended).” |
| Preserve an existing full-cache answer when `--no-full-cache-answer` is used. | The flag stops new generation; it does not delete data. | Remove or ignore existing full answers. | User formulation: “we just add full cache dont delete it.” |
| Allow jobs for different tasks to share a run without locks. | Each task writes to its own directory. The jobs must use the same manifest. Same-task overlap and metric write races are unsupported. | Add process locks for every possible overlap. | User formulation: “there should not be overlap in parallel runs ... it is safe to ignore this case.” |

### Metrics and W&B

| Decision | Why | Alternative | Approval |
|---|---|---|---|
| Store absolute score and actual retention for every task and requested ratio. Store relative score when a valid full-cache baseline exists. | These show quality, quality relative to full cache, and achieved compression. | Store only absolute score. | User formulation: requested `test/<task>`, `test/<task>-relative`, and `test/<task>-actual-retention`. |
| Use a 0–100 scale for absolute and relative scores, and 0–1 for actual retention. | This matches the existing score display and the natural retention fraction. | Use one scale for everything. | Approved plan. |
| Use requested retention on the x-axis. | It is the controlled experiment setting. | Plot against actual retention. | User formulation: requested retention on x and actual retention on y. |
| Average actual retention equally across examples. | Benchmark examples already receive equal weight. | Weight examples by token count. | User answer: “Mean per example (Recommended).” |
| Compute relative score as task score divided by that task's full-cache score. | This matches the repository's existing relative metric. | Average per-example relative scores. | Approved plan. |
| Omit relative and full-cache points when the full-cache baseline is incomplete or zero. | The relative value would be undefined or misleading. | Fail all metric generation. | User answer: “Omit relative curve (Recommended).” |
| Keep the old cross-task aggregate in `metrics.json`, but do not upload it. | Existing local information remains available without adding an unwanted W&B chart. | Remove it or upload another chart. | Approved plan. |
| Rebuild `metrics.json` immediately after each task. | Finished task metrics become available before later tasks finish. All saved tasks remain included. | Wait for the whole evaluation command. | User formulation: “calculate the metrics immediately and write metrics.json.” |
| Upload one task to W&B as soon as that task and its ratios are complete. | A point still represents a full benchmark. It does not wait for other tasks. | Wait for the whole evaluation command or upload partial scores. | User formulation: “after we have outputs for an entire benchmark, calculate the metrics immediately and ... upload to w&b.” |
| Treat 101 loaded SQuAD contexts as complete. | The current repository loader includes indices `0` through `100`. | Treat 100 contexts as complete. | Approved plan. |
| Add the full-cache point at retention `1.0` only when the baseline is complete and nonzero. | The chart does not show a partial or undefined baseline. | Never add the point or add partial points. | Approved plan. |
| Record example count, dataset size, completeness, and mixed full-answer coverage in `metrics.json`. | Partial local results remain readable without being uploaded. | Store scores only. | Approved plan. |
| Use local result files to skip evaluation. Use W&B only to avoid uploading a metric twice. | W&B aggregates cannot prove which examples were evaluated locally. | Use W&B history as evaluation progress. | User clarification and approved plan. |
| Skip matching W&B values, upload missing values, and fail on a different value or duplicate remote value. | A retry does not rewrite training history silently. | Append a second value or overwrite the old one. | User formulation: “if the local metric value and w&b metric value matches (then if not we fail).” |
| Store the checkpoint's training W&B run ID in the manifest. | Metric retries find the training run without loading the checkpoint again. The checkpoint path is still checked separately. | Load the checkpoint during every retry or ask for a run ID. | User review change. |
| Require `--log-to-wandb` and an explicit project. Keep entity optional. | Upload is deliberate and has a known project. | Upload automatically. | User answer: separate flag and required project. |
| Do no W&B work without `--log-to-wandb`. | Local parsing remains offline. | Always inspect W&B. | Approved plan. |
| Keep local files and fail the job if W&B upload fails. | Expensive evaluation work remains available while automation reports failure. | Delete results or report success. | User answer: “Keep files, fail job.” |
| Allow metric parsing and W&B retry without loading the LLM. | An upload retry should not repeat evaluation. | Allow upload only inside the evaluator. | Approved plan. |

### Delivery scope

| Decision | Why | Alternative | Approval |
|---|---|---|---|
| Use Ponytail full and an isolated worktree and branch. | Keep the change small and reviewable. | Work directly on `main`. | User formulation: “work in an isolated worktree and branch” and “use the ponytail skill.” |
| Open a PR to `main`, but do not merge or submit cluster jobs. | The user will review and run jobs. | Merge or submit automatically. | Approved plan. |
| Leave `eval.py`, training, checkpoints, requirements, and old result files unchanged. | This PR changes only graph-evaluation storage and reporting. | Migrate every evaluation path. | Approved plan. |
| Do not print the Git commit in evaluation Slurm logs. | The run does not use that value. | Keep it as diagnostic text. | User formulation: “No need for ‘Print the Git commit in the Slurm log only.’” |

## Under review

These choices are in the current code. They are not treated as settled after
the latest review comments.

| Choice | What the current code does | Why it was added | Simpler alternative | Source |
|---|---|---|---|---|
| Stored JSON validation | The reader checks the question keys, ratio entries, and answer values that it uses. | Resume and metrics should not use a broken result file. | Trust every JSON file written by evaluation. | Failing corrupt results was in the approved plan. The exact checks are **implementation decisions; questioned by user.** |
| Output filename spelling | The reader accepts `0.json` for index 0 and rejects another spelling such as `00.json`. | Both names convert to index 0 and could otherwise be counted twice. The evaluator itself never creates `00.json`. | Ignore nonstandard names, or remove this check because the writer controls filenames. | **Implementation decision; questioned by user.** |
| Agreement across question keys | For one context, every question key must contain the same ratios in the same order. At a ratio, actual retention and threshold must match because pruning happened once. The correct answer and full-cache answer cannot change between ratios. Either every question has a full-cache answer or none does. | The parser currently matches entries by list position and treats one saved ratio as complete for all questions in that context. | Store shared ratio data once at example level and store full answers once per question. This is a larger output-format change. | **Implementation decision; questioned by user.** |
| Full answer copied into ratio rows | The existing JSON shape stores `full__` inside every ratio entry. When a full answer is generated later, the code copies it into every ratio row for that question key. | The new run reader uses the first row, while the old parser effectively uses the last row. Copying the value keeps both paths consistent. This does not rerun the model. | Store one full answer per question outside the ratio list. This changes the result schema and parser. | **Implementation decision; questioned by user.** |
| Atomic JSON replacement | The code writes a complete temporary file beside the target and replaces the old file in one operation. `fsync` asks the OS to flush the temporary file first. | A cancellation during saving should leave the old complete JSON or the new complete JSON, not half a file. This is crash protection, not a lock. | Keep temporary-file replacement but remove `fsync`, or write directly and accept possible truncation. | Atomic saving was in the approved plan. `fsync` is an **implementation decision. Both are questioned by user.** |

## Other implementation decisions

These choices have not been questioned. They remain implementation decisions,
not user requirements.

### Opening and evaluating a run

| Decision | What | Why | Alternative | Source |
|---|---|---|---|---|
| Restrict run and task names to one path component. | Names with path separators, or names equal to `.` or `..`, are rejected. | Deletion and output paths stay inside the intended run directory. | Allow nested names and add path-containment checks. | **Implementation decision.** |
| Preserve the old evaluator path when `--run-dir` is absent. | Direct legacy commands still use the old result saver. | This PR does not break existing evaluation commands. | Make the new run store mandatory. | **Implementation decision.** |
| Open and validate the run before loading the LLM. | Manifest errors stop the command before GPU model loading. | This avoids expensive setup for a run that cannot resume. | Load the model first. | **Implementation decision.** |
| Check saved work before prefill. | A fully cached example does no tokenization, prefill, mixer scoring, or generation. | Resume skips the expensive work, not only the final save. | Prefill before checking. | **Implementation decision.** |
| Skip hidden states and mixer scoring when only a full-cache answer is missing. | The evaluator builds only the KV cache needed for full-cache generation. | Mixer scores are not used without pruning. | Run the full scoring path anyway. | **Implementation decision.** |
| Finish every question for a ratio before saving it. | One save contains all question keys for that context and ratio. | A saved ratio means the example is complete at that ratio. | Save each question separately. | **Implementation decision.** |
| Reuse an existing full-cache answer when adding ratios. | New ratio rows receive the stored answer without new full-cache generation. | Resume performs only missing model work. | Regenerate it for every ratio. | **Implementation decision.** |
| Keep the first copy of a repeated CLI ratio. | Repeated entries are removed. The first position stays. | The user-provided order stays stable. | Sort ratios or keep the last copy. | **Implementation decision.** |

### Metrics and W&B code

| Decision | What | Why | Alternative | Source |
|---|---|---|---|---|
| Use the run reader for both resume and metrics parsing. | Both paths interpret an output file in one place. | The two paths cannot drift to different meanings. | Write a second parser. | **Implementation decision.** The amount of validation is under review above. |
| Keep full numeric precision in `metrics.json`. | Only terminal output is rounded. | Later tools receive the original computed values. | Round stored values for display. | **Implementation decision.** |
| Divide aggregate task score by aggregate full-cache score. | Relative performance is computed once per task and ratio. | This matches the old repository parser. | Average per-example ratios. | **Implementation decision.** |
| Resolve a missing W&B entity from the account default. | The user may omit `--wandb-entity`. | The W&B account already has a default entity. | Require an entity every time. | **Implementation decision.** |
| Require the target training run to be finished. | Upload stops if training is still active. | Evaluation does not write into an active training history. | Allow upload while training is running. | **Implementation decision.** |
| Compare all relevant W&B values before uploading any. | Matching values are skipped, missing values are added, and conflicts or duplicate remote values fail before a write. Values within `1e-9` count as equal. A single remote point outside the local result set is left alone; duplicate remote points still fail. | A failed retry cannot upload some new values before discovering an existing conflict. | Compare and upload one value at a time, or allow overwrites. | **Implementation decision.** |
| Upload one row per retention x-value and define it as the x-axis for every task curve. | Requested ratios get rows. A complete full-cache baseline can add a row at `1.0`. Each row contains only missing absolute, relative, or actual-retention values. | Partial upload retries remain idempotent and W&B draws the requested curves. | Re-upload complete rows or rely on an implicit W&B x-axis. | **Implementation decision.** |
| Disable W&B system monitoring for the short upload process. | The upload process records no machine statistics. | Post-processing adds only test curves and no second system panel. | Use normal W&B monitoring. | **Implementation decision.** |
| Close the W&B writer normally even if upload code raises, then fail the evaluation job. | The shell job fails, but the W&B training run keeps its finished status. | The evaluation failure should not relabel completed training. Local files remain saved. | Mark the training run failed. | **Implementation decision.** |
| Allow a missing W&B run ID for local-only evaluation. | The manifest stores `null`. Local evaluation continues. | Checkpoints trained without W&B can still be evaluated. Upload still requires an ID. | Reject every such checkpoint. | **Implementation decision.** |
| Fail early when W&B upload has no run ID. | The evaluator checks after loading the checkpoint and before loading the LLM. | An unuploadable benchmark does not consume GPU time. | Finish evaluation and fail during upload. | **Implementation decision.** |
| Reuse known dataset sizes while rebuilding metrics. | Existing task sizes come from `metrics.json`. The just-finished task uses its loaded dataset length. | This avoids loading those datasets again. | Reload every dataset after every task. | **Implementation decision.** |
| Rebuild metrics from outputs during a direct retry. | Existing metrics provide known dataset sizes. Missing sizes load from the task dataset. GSM uses the repo's fixed size of 100. | A stopped finalization can be retried without the LLM. | Require an existing `metrics.json`. | **Implementation decision.** The user allowed loading data to obtain its size. |

### Shell behavior

| Decision | What | Why | Alternative | Source |
|---|---|---|---|---|
| Derive the result directory and Slurm job name from one run name in the helper. | Users do not repeat the run path or job name. The result mode remains a separate flag. The helper keeps the evaluator's default graph tag. Arguments after `--` go to Python; batch-only flags stay before it. | This keeps one source for the evaluation run name. | Require users to pass every derived value. | **Implementation decision.** |
| Let Python perform overwrite. | The same code checks and deletes the exact run directory. | Avoid duplicate deletion logic in the shell helper. | Delete in Bash before submission. | **Implementation decision.** |
| Finalize metrics inside the evaluator after each concrete task. | The evaluator calls the finalizer after the task loop. | It knows when the task ends and already has its dataset size. | Run the parser once at the end of the Slurm job. | **Implementation decision.** It implements the user's immediate-finalization request. |
| Forward W&B options from the batch script to the evaluator. | Batch W&B flags become evaluator arguments. | The evaluator now performs the upload itself. | Keep a second parser command in the batch script. | **Implementation decision.** |
| Put temporary W&B SDK files in a temporary directory. | Python removes the temporary directory after upload. | W&B does not create another durable results location or dirty the repository. | Keep a `wandb/` directory beside results. | **Implementation decision.** |

Small code details that do not change the pipeline are intentionally omitted.
Examples are the checkpoint read chunk size, import timing, and result iteration
order.

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

Keeping parser and W&B tests independent of Torch is an **implementation
decision**. Keeping the old training and architecture audit, with a link to this
new evaluation audit, is another **implementation decision**.
