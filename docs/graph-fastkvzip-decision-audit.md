# Implicit Whole-Context Graph FastKVzip: Decision Audit

> Evaluation storage, resume, and W&B decisions in this historical audit are
> superseded by [Evaluation Resume and W&B Decision Audit](evaluation-wandb-decisions.md).
>
> The normalization sections below describe the original BatchNorm baseline.
> The current mixer also supports no normalization and the compact GraNoLa
> adaptation summarized in the [prefill README](../prefill/README.md).

This is the review guide for the original implicit mixer baseline. Its
decisions still describe current code except where the superseding notes above
say otherwise.

Each row states the decision, why it exists, alternatives, and its strongest
conversation source. Source order is: user formulation, user agreement,
approved plan, inherited behavior, then implementation-only.

## Implementation-led decisions (review these first)

These were selected from PyTorch behavior, storage limits, or pilot evidence.
They were not architectural choices you explicitly prescribed.

- **`PyTorch inference-tensor requirement`**: create normal CPU hidden tensors
  before student training.
- **`FP32 master parameters`**, **`Chan/Welford arithmetic`**, and
  **`no-grad prepare, proxy backward replay`**: keep mixed-precision streamed
  training stable without retaining a full context-width autograd graph.
- **`global mixer BCE normalization`** and **`CPU offload between gate slices`**:
  make graph/token microbatching exact while keeping two-phase GPU memory bounded.
- **`batch gate computation across each graph microbatch`**: remove serial
  per-head gate calls while preserving independent checkpoint weights.
- **`cache before dataset/wrapper/prefill`** and **`temp file plus hard link`**:
  skip all teacher work on a hit and publish cache files without overwriting.
- **`release the teacher after a complete cache hit`**: free the base LLM on a
  fully warm run, while keeping partial-cache generation simple.
- **`cadence is not a checkpoint invariant`**: permit a safe resume with a
  different operational save/evaluation frequency.
- **`terminal last checkpoint`**, **`timing excludes teacher work`**, and
  **`finally-based hidden-cache release`**: preserve useful recovery state and
  avoid misleading measurements or retained evaluation memory.
- **`synchronized evaluation timing`**, **`per-example peak allocated memory`**,
  and **`failure diagnostic replay`**: keep the task bar useful without losing
  actionable failure evidence.
- **`checkpoint shape/dtype validation before LLM construction`**: reject a
  bad evaluation checkpoint before allocating the model.
- **`NumPy indices`** and **`low-precision mixer gradient`**: pilot fixes for
  the actual cluster dataset and half/bfloat16 backward path.

Search the bold phrases below for the full decision, rationale, alternatives,
and source.

## Complete pipeline

For one dataset context:

1. The teacher pre-fills prefix plus context in LLM prefill chunks.
2. The retain cache captures one CPU hidden-state tensor per transformer layer.
3. KVzip reconstruction produces teacher labels for context positions.
4. The trainer releases the teacher KV cache and keeps CPU hidden tensors,
   labels, token IDs, prefix IDs, and context identity.
5. Each layer/KV-head graph makes a whole-context implicit mixer result.
6. The matching FastKVzip gate scores the original hidden state plus that
   residual.
7. Joint mode performs one gate and one mixer update per context. Two-phase
   mode performs gate token-slice updates followed by one mixer update.
8. Evaluation restores the exact prefix and prefill chunk, scores only
   context positions, applies the existing local-window protection, clears
   hidden states, then reuses the existing prune/generate loop.

Main source files:

- [implicit mixer and gate adapter](../prefill/graph/model.py)
- [streamed training and checkpoints](../prefill/graph/training.py)
- [checkpoint reconstruction and evaluation scoring](../prefill/graph/evaluation.py)
- [CLI, teacher generation, cache, and W&B](../prefill/train_graph.py)
- [standalone evaluation entry point](../prefill/eval_graph.py)

## Terms and shapes

| Name | Shape | Meaning |
|---|---:|---|
| P | scalar | prefix/system/start-of-turn token count |
| T | scalar | scored context token count |
| L, H | scalar | transformer layers and KV heads |
| D, C | scalar | hidden size and graph dimension |
| M | scalar | complete graphs in a graph microbatch |
| X | conceptually [M,T,D] | selected layer/head hidden states |
| Y1, Y2 | [M,T,C] | low-rank projections |
| S | [M,C,C] | Gram contraction |
| scores | [L,1,H,T] | cache importance scores |

The [M,T,D] form is mathematical. Production retains Y1[M,T,C], small
contractions/statistics, and one token slice of hidden-width work. It never
retains a full mixer output or a T-by-T adjacency.

## Scope and integration

| Decision | Why | Alternatives | Source |
|---|---|---|---|
| Use an implicit low-rank mixer instead of a materialized token graph. | Removes expensive neighbor search and avoids a T-by-T adjacency. | Hard k-NN graph or dense adjacency. | User formulation: “we don't actually need to materialize the adjacency matrix.” |
| Keep the FastKVzip gate unchanged. | Isolates the new experiment and preserves released/local gate checkpoints. | Replace the gate. | Approved plan: keep the FastKVzip gate. |
| Keep eval_graph.py standalone and leave eval.py unchanged. | Baseline evaluation stays easy to review. | Add branches to eval.py. | User formulation: “What if we have a brand new script?” |
| Keep the existing dataset split/cursor. | Runs and resumes stay comparable. | New data runner. | Approved plan. |
| Keep production dependencies limited to the implicit mixer path. | The implementation needs no topology-builder runtime. | Carry an unused graph-library dependency. | Approved plan. |

## Teacher context, prefix, and labels

| Decision | Why | Alternatives | Source |
|---|---|---|---|
| Teacher loads with gate_path_or_name empty. | Labels are KVzip reconstruction labels, not a released-gate score. | Teacher with fastkvzip gate. | Approved plan. |
| Score/train context positions only. | Prefix/system/turn tokens must stay protected. | Score all prefill tokens. | User agreement that train/eval hidden states must match. |
| Store exact prefix IDs and prefill chunk in checkpoints. | Both can affect hidden states. | Rebuild a fresh prefix or choose an eval chunk. | User formulation: “if we change the system prompt or prefix in evaluation things might not work the same.” |
| Capture ordinary CPU hidden tensors outside inference mode. | Mixer parameter gradients need normal tensor inputs. | Clone later or retain GPU activations. | Implementation-only: PyTorch inference-tensor requirement. |
| Transfer captured hidden storage into TeacherExample without a second full clone. | Avoids doubling the largest CPU tensor. | Clone all hidden states. | Approved plan: transfer ownership. |
| Normalize scores, token IDs, and prefix IDs to CPU. | They are small and make cache data device-independent. | Keep them on teacher GPU. | Implementation-only. |

The extraction function slices kv.start_idx:kv.end_idx. Prefix data can remain in
the backing CPU tensor, but it is not a training target.

## Implicit mixer architecture

For flattened graph ID g = layer times H plus head:

    Y1 = X W1[g]
    Y2 = X W2[g]
    S  = Y1 transpose Y2 / T       # or unscaled
    P  = Y1 S W[g]
    Z  = Normalize_context_g(P)
    f  = LeakyReLU(gamma[g] * Z + beta[g])
    X' = X + alpha[g] * f
    score = matching FastKVzip gate head(X')

| Decision | Why | Alternatives | Source |
|---|---|---|---|
| Every layer/KV head owns W1, W2, W, gamma, beta, and alpha. | Each head can learn a different context relation. | Share by head or layer. | User formulation: “we should have different weights for every head.” |
| Use the corrected Y1(Y1 transpose Y2)W formula. | Final W maps C back to D. | Earlier incompatible annotation. | User formulation: “a different W there that is c x d.” |
| Pack W1/W2 into one D-to-2C projection. | One batched projection is faster; slices stay independent. | Two projection modules. | Approved plan. |
| Keep FP32 master parameters for gates and mixer when compute dtype is FP16/BF16. | Optimizer updates and checkpoint tensors remain stable while activations use the requested fast dtype. | Train parameters directly in FP16/BF16. | Implementation-only. |
| Use a headwise gate adapter instead of materializing one full mixed hidden tensor per KV head. | It applies only the matching delta to the matching released gate head. | Change FastKVzip's full gate or duplicate full hidden tensors. | Approved plan. |
| Batch gate projection and scoring across each graph microbatch; group only RMSNorm by transformer layer. | It removes serial per-head gate calls while reusing the exact layer norm modules. Stacked parameter slices still use every head's independent checkpoint weights. | Call the gate once per head. | Implementation-only: warm-cache performance evidence. |
| Use bias-free Kaiming-uniform W1/W2/W. | Exact requested parameterization. | Bias, Xavier, or zero initialization. | Approved plan. |
| Default graph dimension is 32. | Requested latent default. | Smaller/larger default. | User formulation: graph dim 32. |
| Default Gram normalization is token-count; allow none. | Keeps scale stable as T changes. | Always unscaled. | Approved plan. |
| Normalize before LeakyReLU. | The requested correction applies current-context BatchNorm to the linear message, then applies the affine activation. | LeakyReLU before BatchNorm. | User formulation: “first normalize then leaky relu.” |
| Use LeakyReLU slope 0.01. | Negative messages survive. | ReLU, GELU, learned slope. | Approved plan. |
| Initialize gamma=1, beta=0, alpha=0.1; alpha is unconstrained. | Residual starts small and can learn either sign. | Zero, sigmoid-constrained, or per-token alpha. | Approved plan. |
| Keep signed weights and implicit self-connections. | Y1Y1-transpose may be signed and diagonal entries are self messages. | Clamp, remove diagonal, normalize rows. | Approved plan. |

## Context BatchNorm

| Decision | Why | Alternatives | Source |
|---|---|---|---|
| Compute independent stats per layer/head across that context's T tokens. | No layer/head mixes statistics. | Batch stats across graphs. | Approved plan. |
| Use population variance, divide by T. | Explicit requested behavior. | Divide by T-1. | Approved plan. |
| Add 1e-5 inside square root. | Singleton/constant contexts remain finite. | No epsilon or another epsilon. | Approved plan. |
| Keep no running averages; recompute at train and eval. | Current context must define normalization. | Standard BatchNorm running state. | Approved plan. |
| Apply learnable per-feature gamma/beta after normalization and before LeakyReLU. | This is the affine part of Context BatchNorm before the corrected activation. | Affine transform after LeakyReLU or none. | User formulation: “first normalize then leaky relu.” |
| Accumulate Gram/statistics in FP32, but preserve FP64 in numerical tests. | Stable half/bfloat16 production and exact float64 verification. | Always model dtype. | Implementation-only. |
| Merge streamed moments with Chan/Welford arithmetic. | Exact population stats without retaining P[M,T,D]. | Materialize all preactivations. | Implementation-only. |

## Chunking, memory, and exact gradients

| Decision | Why | Alternatives | Source |
|---|---|---|---|
| Prefill chunks do not split the graph. | Later LLM chunks attend prior KV cache; the result is one context. | Independently prefill subcontexts. | User agreement: hidden states should be identical in train/eval. |
| Graph microbatch auto equals H. | One complete layer is default. | One graph or all graphs. | Approved plan. |
| Explicit graph microbatch must be 1 through L times H. | Invalid graph batches fail before teacher generation. | Clamp silently. | Approved plan. |
| Token microbatch defaults to 1,000. | It bounds temporary hidden-width work; joint/mixer gradients stay invariant, while two-phase gate updates intentionally use it as their update batch. | Full-context hidden-width ops. | User question: “token microbatch size is purely efficiency right?” |
| Retain Y1[M,T,C], not preactivations/residuals. | C is much smaller than D. | Retain full P or delta. | Approved plan. |
| Use two streamed loss passes for exact BatchNorm gradient. | BatchNorm couples all T tokens. | Treat token chunks as independent BN batches. | Approved plan. |
| Use no-grad prepare, proxy backward replay. | The forward retains compact values only; the backward rebuilds exactly the local autograd pieces it needs. | Retain the complete forward autograd graph. | Implementation-only. |
| Use global mixer BCE normalization over the full L-times-H-times-T score count. | Graph microbatch size and a short final token slice cannot change the gradient scale. | Average each graph microbatch independently. | Implementation-only. |
| Use CPU offload between gate slices for frozen-mixer prepared state. | Gate phase reuses Y1, Gram, kernel, and normalization without holding them on GPU. | Retain them on GPU or recompute per slice. | Implementation-only. |
| Backpropagate complete graph gradients before an optimizer step. | Mixer gets one update per context, not T/1000 updates. | Update per token chunk. | User question about whether the graph update happens T/1000 times, then approved clarification. |
| Verify staged float64 gradients against ordinary full autograd. | Tests the streamed algebra. | Only check finite loss. | Approved plan. |

Autograd first passes the loss gradient through alpha, LeakyReLU, gamma, and
beta. The implemented population-BatchNorm backward then maps it from Z to P:

    dP = invstd * (dZ - mean(dZ) - Z * mean(dZ * Z))

Z is normalized P. Training then maps the full Gram gradient back to Y1/Y2
and back through the packed input projection.

## Training modes and optimizers

| Decision | Why | Alternatives | Source |
|---|---|---|---|
| Joint mode is default. | One gate and one mixer update per whole context. | Two-phase default. | Approved plan. |
| Joint mode permits different gate/mixer LRs and schedulers. | A pretrained gate can use lower LR than a new mixer. | Require equality/copy settings. | User formulation: “why can't we have separate learning rates … for the mixer and for the gate?” |
| Defaults are gate LR 1e-4 and mixer LR 1e-3. | Mixer starts from scratch. | One shared LR. | Approved plan. |
| Support PyTorch scheduler names plus JSON kwargs. | Reuses standard scheduler behavior. | Custom scheduler language. | Approved plan. |
| Default both schedulers to none. | Fixed learning rates are the simplest reproducible baseline; schedules are opt-in per optimizer. | Implicit decay schedule. | Implementation-only. |
| Step normal schedulers after their optimizer; plateau after validation. | Matches PyTorch semantics. | Step before optimizer. | Approved plan. |
| Two-phase remains optional. | Gate has ceil(T/1000) shuffled updates; mixer has one context update. | Remove staged mode. | Approved plan. |
| A cadence step means one completed training context. | Joint and two-phase have different numbers of inner optimizer calls. | Count raw optimizer calls or token slices. | User question: “After each step?” |
| Default checkpoint and validation cadence is every epoch. | Avoids frequent checkpoint I/O while retaining one recovery point per epoch. | Save every training context or validation context. | User formulation: “default save should be every epoch.” |
| Gate phase sees current mixed inputs. | Prevents raw-only gate preference. | Train it on raw hidden states. | User concern that the gate may “prefer the raw hidden states.” |
| AdamW decays W1/W2/W only; alpha/gamma/beta get zero decay. | Projection and affine parameters need different regularization. | One decay group. | Approved plan. |
| Existing gate parameter grouping is unchanged. | Preserves FastKVzip behavior. | Repartition gate weights. | Approved plan. |

The new preferred CLI names say mixer. Existing graph-LR spellings are aliases
for command compatibility. Checkpoints use the unambiguous mixer names.

## Teacher cache and resume

| Decision | Why | Alternatives | Source |
|---|---|---|---|
| Teacher cache directory is optional. | Missing option preserves online-only generation. | Require disk cache. | Approved plan. |
| Cache one file per dataset name/example index. | Partial caches and resume work naturally. | One monolithic file. | Approved plan. |
| Store hidden states, scores, token IDs, prefix IDs, identity, length, model ID, and prefill chunk. | A cache hit can be validated before prefill. | Store activations only. | Approved plan. |
| Cache before dataset/wrapper/prefill construction. | A hit avoids all teacher work. | Check later. | Implementation-only. |
| Release the teacher after student construction when every configured train/validation cache file exists. Rebuild it lazily only if a file later goes missing; a partial cache keeps it loaded. | A fully warm run does not retain the base LLM in GPU memory. Partial-cache generation avoids repeated model loads. | Retain the teacher for the whole run or construct the student from configuration without first loading the LLM. | Implementation-only: warm-cache memory evidence. |
| Fail on corrupt/incompatible cache. | Never silently train on wrong teacher data. | Regenerate/overwrite automatically. | Approved plan. |
| Publish cache with temp file plus hard link, never replacement. | Atomic final creation without overwriting an existing cache. | os.replace. | Implementation-only. |
| Cache path is not a checkpoint resume invariant. | A resume can use a different scratch path. | Store/compare it in checkpoint config. | Approved plan. |
| Warm the shared grid cache with the valid one-epoch random/trainable run, then start the other eight runs with an `afterok` dependency. | Concurrent cache misses deliberately fail rather than overwrite; one completed epoch generates all training and validation cache files. | Separate cache-only job, isolated caches, or concurrent cache writers. | User formulation: “we can use [shared storage] to generate the data once and save it on disk.” |
| Put fixed-grid checkpoints in `graph_checkpoints/<meaningful-run-name>`. | It preserves the training CLI's default output root while isolating all nine runs. | One shared output directory or the generic Slurm `runs/graph` root. | User formulation: “save checkpoints to the default output dir with meaningful experiment names.” |
| Store the fixed activation order in checkpoint configuration. | Training and standalone evaluation cannot silently disagree about normalization and activation order. | Infer it only from the installed code. | Implementation-only. |
| Cadence is not a checkpoint invariant. | Save/evaluation frequency can change on resume without changing the trained model or optimizer configuration. | Require matching cadence settings. | Implementation-only. |
| Save `last.pt` at the configured training-context or epoch cadence, after any due full validation sweep. | A plateau scheduler's state is included while no checkpoint is written per held-out context. | Save after every validation context or before validation. | User request for save strategy/every controls. |
| Write a terminal last checkpoint when `--max-contexts` or a run boundary stops before the selected cadence. | A one-context pilot and short interrupted run still have a resume point. | Discard the most recent unsaved training context. | Implementation-only. |
| Save `best.pt` only after a completed validation sweep improves the mean BCE, unless `--no-save-best` is set. | Normal runs retain held-out selection; storage-limited grids can keep only the repeatedly replaced `last.pt`. | Delete `best.pt` after writing it or disable validation. | User formulation: “save only the last checkpoint every time.” |
| Load current checkpoint configuration and state strictly. | Architecture mismatches fail at load time. | Partial or permissive loading. | Approved plan. |

Each checkpoint includes mixer/gate state, both optimizer/scheduler states,
architecture config, model ID, prefix IDs, prefill chunk, cursor, RNG state,
and W&B run ID. State loading is strict.

## Evaluation and protection

| Decision | Why | Alternatives | Source |
|---|---|---|---|
| Restore checkpoint prefix IDs and prefill chunk. | Reproduces training hidden-state conditions. | Use current defaults. | Approved plan. |
| Use checkpoint shape/dtype validation before LLM construction. | Configuration, tensor keys, shapes, and dtypes fail before expensive model allocation. | Discover it during/after model load. | Implementation-only. |
| Score only kv.start_idx:kv.end_idx. | Prefix/start-of-turn tokens are not context nodes. | Score entire cache. | Approved plan. |
| Preserve prefix/turn/query/postfix/local-window/generated protection in existing cache code. | Mixer must not redefine cache safety. | New custom pruning code. | Approved plan. |
| Apply existing local-window score override. | Keeps pruning behavior unchanged. | Hard mask or no window. | Approved plan. |
| Also force the complete local window into the final boolean keep mask after thresholding. | A tied maximum-score window can otherwise be entirely dropped by strict pair/layer thresholds or partly dropped by head top-k when its size exceeds the requested budget. One shared post-threshold operation covers every pruning level. | Change every threshold algorithm separately or rely only on tied scores. | User approval: “Yes let's fix both.” |
| Treat the protected window as a minimum retention ratio. | If 4,096 protected tokens exceed the requested budget, retaining all of them is more important than matching that budget. The saved actual ratio reports the final mask honestly. | Drop part of the protected window or reject the ratio. | Consequence of the approved protection requirement. |
| Keep the maximum-score override as well as the hard mask. | When the requested budget is large enough, protected tokens consume that budget instead of being added on top of an unrelated top-k selection. | Remove the override and always increase retention by the full window size. | Implementation-only. |
| Pin `datasets==4.0.0`. | It is the first release that understands the Hub's `List` feature metadata, and an isolated live SQuAD load passed with the full project pins. | Keep 3.6.0 and pin old dataset metadata, bypass the Hub schema with custom Parquet loading, or use a newer untested release. | User approval: “Yes let's fix both.” |
| Fail evaluation before model loading when the active environment does not have `datasets==4.0.0`. | Pulling the repository does not update the reused cluster environment. A fast check gives the exact one-time upgrade command instead of failing near the end of the benchmark suite. | Install packages inside every Slurm job or rely only on documentation. | Implementation-only. |
| Always clear hidden cache after score assignment. | It is the largest temporary. | Retain through generation. | Approved plan. |
| Use finally-based hidden-cache release. | Scoring failures cannot leave the largest evaluation temporary resident. | Clear only after a successful score assignment. | Implementation-only. |
| Generate the full-cache reference answer by default, with `--no-full-cache-answer` to skip it. | Existing results stay compatible, while grids sharing one base model can avoid repeated reference generation. Disabled runs store `full__` as null and still generate every pruned answer. | Always regenerate it, use the ground truth as a false baseline, or add a shared answer cache. | User formulation: “add a flag whether or not to extract a full cache answer from the LLM.” |
| Allow an explicit list of evaluation retention ratios while preserving the original list by default. Use the same list in result parsing. | A run can avoid generations at ratios outside the experiment, and every saved ratio remains measurable. | Always run the five inherited ratios or edit source code per experiment. | User formulation: “I want to try only 10%/20%/30% retention ratios.” |
| Keep `eval_graph.py`, the shared result saver, and `results.parse` unchanged for result organization. Run the evaluator from the project root instead. | The inherited relative paths then make project-root checkpoints work and move the unchanged benchmark-first result tree from `prefill/results` to `results`. | Add a graph-specific saver and `--run-dir` parser mode. | User formulation: “I prefer if possible to not change code and just change the scripts and how I run it”; later confirmation: “we don't change folder structure, just move the results folder.” |
| Submit graph evaluation through `submit_eval_graph.sh`. The helper accepts the run name once, sets the Slurm job name, resolves the checkpoint, and forwards all evaluation flags. | One short command handles resource and path plumbing without duplicating the run name or hiding evaluation options such as microbatch sizes. | Define a shell variable around a direct `sbatch` command or add Python orchestration. | User formulation: “I prefer to have a helper that will submit the sbatch for me ... [and] pass other evaluation flags.” |
| Write evaluation logs as `.slurm/logs/<job-id>-<run-name>.log`. | Job identity and evaluation identity are visible in one predictable filename. | Keep the generic `graph-eval-<job-id>.out` name. | User formulation: “the job log [should] be saved to `.slurm/logs/<job-id>-<job-name>.log`, also the job name should be the evaluation run name.” |
| After successful evaluation, run the unchanged metric parser with the same data, level, and ratios. Print its output in the job log and tee it to `results/metrics/<run-name>.txt`. | Raw answers and the readable aggregate are both durable, while parser failure correctly fails the job. | Parse manually, save metrics only in the log, or change the parser output format. | User formulation: “when the evaluation is finished I want to run the script that gives all the metrics.” |
| Refuse a helper submission when matching root results or metrics already exist. | The inherited saver overwrites matching JSON files, so a fresh run name prevents accidental mixing without deletion. | Resume/overwrite existing results or add a new run manifest. | User selection: fail immediately when the evaluation run already exists. |
| Restore evaluation microbatch sizes from the checkpoint by default, but allow efficiency-only overrides. `full` uses the complete current context as one token chunk; `all` uses every layer/head graph as one graph batch. | Evaluation has no backward pass, so it may fit larger batches than training. The explicit keywords avoid editing checkpoint metadata or guessing the longest context. | Always use training sizes, or require large numeric sentinels. | User formulation: “run the evaluation with maximal token microbatch and maximal graph microbatch.” |
| Keep full-context prefill even when full-cache answer generation is disabled. | The checkpoint still needs context hidden states and a complete KV cache for scoring and pruning. | Cache base-model hidden states and KV tensors across jobs. | Implementation-only. |
| Reject a result directory that mixes files with and without full-cache answers. | Otherwise relative performance would compare a full-cache denominator from fewer examples with pruned numerators from every example. | Silently use only the shared subset or report the misleading ratio. | Implementation-only. |

Evaluation calls the existing DataWrapper, evaluator, ratio loop, prune method,
and result saver from the standalone eval_graph.py entry point.

## Logging and observability

| Decision | Why | Alternatives | Source |
|---|---|---|---|
| Log training metrics once per training context, and `validation/bce` once per full validation sweep. | The validation value matches scheduler and best-checkpoint selection. | Log four partial validation BCE values. | User question: “what evaluation metrics are logged to w&b?” |
| Put all optimization losses and learning rates under `train/`. | The dashboard has one training section instead of separate joint/gate/mixer sections. | One namespace per phase. | User formulation: “instead of having joint/ mixer/ and gate/ sections we can just put everything under train/.” |
| Log `train/bce` in joint mode, and `train/gate_bce` plus `train/mixer_bce` in two-phase mode. | The metric name retains the active phase without creating another W&B section. | Overload one BCE key or use phase namespaces. | Implementation of the user direction above. |
| Log one `train/mean_alpha`, rather than one alpha chart per layer/head. | It tracks the learned residual scale without creating hundreds of W&B charts. It is the signed mean of all layer/KV-head alpha scalars after the context update. | Per-head alpha metrics, histograms, or an absolute-mean companion metric. | User formulation: “log alpha to w&b.” |
| Log `train/epoch` after every completed context. | It gives a continuous progress value: `1/34` after the first training context and `1.0` after the first epoch. | Integer epoch only or a batch counter only. | User formulation: “log epoch float every step to w&b.” |
| Log `train/tokens` from scored training contexts only. | It measures real student-data progress and resumes exactly from the checkpoint cursor without counting prefix, validation, layer, head, or microbatch duplication. | Count all model-input tokens or reset the count each epoch. | User formulation: “log accumulated training tokens to w&b.” |
| Show one `tqdm` bar for completed training contexts at terminal position 1. Its postfix mirrors the compact `train/` metrics. | It makes student progress visible while leaving the existing position-0 KVzip prefill bars readable. | Per-token/microbatch bars or no student progress bar. | User formulation: “training logs only show progress of kvzip, it should also show training process with tqdm and log the metrics we log to w&b.” |
| Put phase timing under `timing/`, not `train/`. | Performance measurement is not an optimization loss. | Put timing under `train/`. | User formulation: “forward and backward times into another namespace it does not belong in train/.” |
| Divide every timing value by the example's scored context length T, and name it `*_seconds_per_token`. | It makes contexts of different lengths comparable. | Log total seconds. | User formulation: “timing metrics to be divided by the context length of the example.” |
| Do not emit `gpu/*` peak-memory metrics. | W&B's asynchronous system monitor already reports GPU state, and duplicate charts clutter the run. | Keep PyTorch peak metrics alongside the system monitor. | User formulation: “we also dont really need the gpu section because system reports this.” |
| Keep mean `validation/bce` under `validation/`. | Held-out loss stays visibly separate from updates. | Put it under `train/` or log individual-context BCEs. | Approved plan, refined by the new cadence. |
| Average the four per-context validation BCE values equally. | Each held-out context has equal influence even if their token counts differ. | One token-weighted validation BCE. | Implementation-only. |
| Timing excludes teacher work and validation timing. | Teacher prefill/cache work is not student training cost, and validation charts stay compact. | End-to-end timing or validation timing metrics. | Implementation-only. |
| Use W&B's asynchronous system monitor for GPU utilization. | Avoid synchronous utilization polling in the loop. | Poll nvidia-smi each token. | Approved plan. |
| Do not log delta energy. | Explicitly removed from metric set. | Retain old metric. | Approved plan. |
| Show one real, in-place `tqdm` bar per standalone-evaluation task. Keep its default count, elapsed time, ETA, and iteration rate. | One changing line shows task progress and completion time without producing one permanent line per example. | Print each example, use a custom static status line, or remove progress reporting. | User formulation: “I want everything to be on the same line to avoid repeated logs” and “show the default tqdm iterations per second or seconds per iteration.” |
| Refresh the task bar after each evaluation operation. | `max_tokens` and `max_gpu` can increase before the example finishes. Timing averages change only after the complete example. | Refresh only after each example. | Implementation decision. |
| Show the largest context processed in this task invocation as `max_tokens`. Show running mean ± population standard deviation for completed examples in this invocation. | Peaks and accumulated timing shares are more useful for choosing task-specific microbatches than one example's values. Cached examples have no stored telemetry. | Show the current example or raw phase seconds. | User formulation: “instead of tokens, show max_tokens so far in the task” and requested accumulated averages with “±std.” Population standard deviation is an implementation decision. |
| Divide each example's synchronized phase seconds by its synchronized total time before updating the task averages. | Every example has equal weight, regardless of length or runtime. | Divide summed phase times by summed total time. | User example and implementation decision. |
| Reset CUDA peak statistics once per task and display peak PyTorch-allocated memory over total device memory as `max_gpu`. | The largest task allocation stays visible after the example that caused it. | Reset per example, display reserved memory, or poll external utilization. | User formulation: “I would like to see the gpu peak for the task.” |
| Hide ordinary per-example evaluation output by default and restore it with `--verbose`. Replay captured diagnostics before the traceback if an example fails. | Normal logs stay compact, while verbose review and failures retain the old evidence. | Delete detailed messages, always print them, or suppress failure context. | User formulation: “all these things are saved to the final json so I don't need to see it really in the evaluation logs”; `--verbose` and failure replay from the approved plan. |
| Keep standalone-evaluation result JSON unchanged; progress timings and GPU measurements remain terminal-only. | Logging changes do not alter result compatibility or benchmark parsing. | Add progress telemetry to every result record or change existing answer fields. | User formulation: “saving what actually matters to the json”; explicitly fixed by the approved plan. |

Training CUDA event timing synchronizes only at the context boundary.

The external metric keys are `train/bce`, `train/gate_bce`,
`train/mixer_bce`, `train/mean_alpha`, `train/epoch`, `train/tokens`,
`train/gate_learning_rate`,
`train/mixer_learning_rate`, `validation/bce`, and
`timing/<phase>_{forward,backward}_seconds_per_token`. Internal `graph`
timing is reported as `mixer` because it measures the implicit mixer phase.
`validation/bce` is the mean across the four held-out contexts. Standalone
benchmark evaluation does not log to W&B.

## CLI and operational behavior

| Decision | Why | Alternatives | Source |
|---|---|---|---|
| Keep gate-dim terminology. | It describes FastKVzip dimension directly. | Generic dim. | User formulation: “I would like to have --gate-dim instead of --dim.” |
| Keep gate-sink. | It reconstructs FastKVzip learned base keys. | Infer only from checkpoint. | Inherited behavior. |
| Remove topology-builder and residual-B initialization options. | They no longer describe this model. | Ignore deprecated options. | Approved plan. |
| Accept released fastkvzip and local gate checkpoints. | Mixer can start from released or local gate. | Random-only gate. | Approved plan. |
| Reject freeze-gate without a gate/resume checkpoint. | Freezing a random gate is likely accidental. | Allow silently. | Inherited behavior retained. |

Gate sink is not cache prefix protection. It is the count of learned FastKVzip
base keys in gate.k_base.

## Verification

Focused tests cover:

- implicit multiplication against explicit dense algebra;
- normalization-before-activation dense algebra and checkpoint metadata;
- packed W1/W2 independence;
- layer/head BatchNorm isolation, singleton/constant contexts, scale modes, alpha;
- no token-by-token matrix output;
- batched gate adapter output and gradient parity with serial headwise calls;
- graph/token microbatch invariance for mixer/joint training;
- streamed float64 gradients versus full autograd;
- independent joint optimizer/scheduler settings and AdamW groups;
- compact W&B metric names, alpha/progress/token logging, per-token timing normalization, and no `gpu/*` metrics;
- quiet/verbose standalone-evaluation progress, failure replay, phase timings,
  per-example GPU peaks, and unchanged result JSON;
- save/evaluation cadence, complete validation sweeps, and validation-mean logging;
- teacher-cache creation/reuse/partial/mismatch/corruption;
- current checkpoint save/load;
- context-only evaluation scoring, local window, and hidden-cache release;
- no unused topology-library production imports.

One Qwen3-8B Slurm context completed successfully after the PR was pushed.

## Slurm pilot

| Decision | Why | Alternatives | Source |
|---|---|---|---|
| Push PR before pilot so review starts first. | Cluster time is used on reviewable code. | Submit before push. | User formulation: “after the PR will be ready and pushed.” |
| Run sres immediately before submission. | GPU availability is live state. | Use earlier status. | User formulation: “dont check with sres now.” |
| Prefer rtx_pro_6000:1; choose rtx_6000:1 when live availability is better. | Keeps the preferred GPU while avoiding an unnecessarily long queue. | Hard-code one type. | User formulation: “we can request rtx 6000 if there are more available.” |
| Pilot uses one hour, 60G RAM, 40G scratch. Later cached full run uses 600G scratch after recheck. | Pilot measures actual limits before scale-up. | Full resources immediately. | Approved plan. |
| Teacher cache uses node-local scratch; checkpoints/logs use durable shared storage. | Cache is disposable; checkpoints/logs are not. | Put all data in scratch. | Approved plan. |

Slurm parses SBATCH lines before shell commands. The live GPU choice must be
made with sbatch --gpus=selected-type:1 after inspecting sres, not inside a
static batch script.

## Pilot-resolved details

| Decision | Why | Alternatives | Source |
|---|---|---|---|
| Convert FineWeb's selected NumPy indices to Python integers before dataset lookup. | The cluster's dataset version rejects NumPy scalar indices. | Cast at every lookup or rely on implicit conversion. | Implementation-only: first pilot failure. |
| Finish the W&B run with exit code 1 on any exception and 0 only after the full loop completes. | Failed training no longer appears successful in the dashboard. | Always call `finish()` with its default status. | User formulation: “the w&b run should be marked as failed or something because now it looks like success.” |
| Cast the staged low-precision mixer gradient to the loss working dtype before adding it to the FP32 accumulator. | Half/bfloat16 pilots can complete the exact streamed backward without a dtype-copy failure. | Force all staged buffers to model precision or run the whole path in FP32. | Implementation-only: second pilot failure. |
| Use online W&B when `--wandb-mode online`; offline and disabled modes do not log in. | The supplied API key can sync the pilot and training metrics immediately. | Always use offline mode. | User question: “why not use w&b in online mode? I have an api key”. |

The final one-context Qwen3-8B pilot completed after these fixes. Its role was
to validate the actual cluster path; resource limits for a full run still need
the required live `sres` check.

## Deliberate non-decisions

- Baseline eval.py is unchanged.
- No parameter watching, gradient histograms, heatmaps, or extra ablation passes.
- No k-NN topology or T-by-T adjacency is materialized.
- Teacher-cache files are never automatically deleted or overwritten.
