## Prefill-Intensive Tasks

### Reproducing Benchmark Results
```bash
python -B eval_chunk.py -g fastkvzip -m $MODEL_ID -d all 
```
- Results will be saved at the ```./prefill/results``` folder. 
- We provide the implementation of other baselines compared in our paper. Please refer to `run.sh`.
- Available data names are listed in `data/load.py`. For MRCR, please run `eval_chunk_mrcr.py`.
- We release gates for the following ```$MODEL_ID```:
    - Qwen/Qwen2.5-{7,14}B-Instruct-1M 
    - Qwen/Qwen3-{8,14}B
    - Qwen/Qwen3-8B-FP8
    - Qwen/Qwen3-4B-Instruct-2507
    - google/gemma-3-12b-it

> [!Note]  
> - In our experiments, we use `--kv_type retain`, which preserves the full KV cache in memory while performing attention over a reduced KV cache via subsampling, following KVzip.
> - For improved speed and lower peak memory usage, use `--kv_type evict`. This option may cause marginal differences in prediction results due to GPU numerical variability.

To get task scores,
```bash
python -B -m results.parse -m qwen2.5-7b-instruct-1m_fastkvzip_chunk16k_w4096 -d all
```
- Please set the folder name for the method using `-m`, as shown above.
- See `./prefill/results/parse.py` for more details.

### Example-Level Analysis
- To check the detailed changes in predictions induced by KV eviction, run
```python
python -B test.py --kv_type evict -g fastkvzip -d scbench_kv
```

### Whole-Context Graph FastKVzip

Run these commands from `prefill/`. Actual training and evaluation require a CUDA GPU and a compatible FlashAttention installation. The graph path currently supports ordinary decoder hidden caches such as Qwen's; hybrid/static cache layouts such as Gemma 3's are not supported.

For the cluster setup, pilot workflow, reusable `sbatch` scripts, and
experiment controls, see [the experiment guide](../docs/graph-fastkvzip-experiments.md).
See [the evaluation status](../docs/evaluation-status.md) for the verified
protocol and current recommendation.

The default is joint training: one whole-context gate update and one
whole-context implicit-mixer update. The released FastKVzip gate is optional
but recommended for the first run:

```bash
python -B train_graph.py \
  --model "$MODEL_ID" \
  --gate-checkpoint fastkvzip \
  --teacher-cache-dir "$TMPDIR/teacher-cache" \
  --output-dir graph_checkpoints/joint
```

Use --training-mode two-phase when the gate should receive shuffled
1,000-token updates before one frozen-gate mixer update:

```bash
python -B train_graph.py \
  --model "$MODEL_ID" \
  --gate-checkpoint fastkvzip \
  --training-mode two-phase \
  --output-dir graph_checkpoints/two-phase
```

The implicit mixer is:

    Y1 = X W1
    Y2 = X W2
    S = Y1 transpose Y2 / T
    X' = X + alpha * LeakyReLU(ContextBatchNorm(Y1 S W))

Every layer/KV head has independent weights. It never materializes a
token-by-token adjacency matrix. Main controls are graph-dim (default 32),
gram-normalization, leaky-relu-slope, alpha-init, graph-microbatch-size, and
token-microbatch-size. Checkpoint/validation controls are save-strategy,
save-every, save-best, eval-strategy, and eval-every.

For more throughput, increase token-microbatch-size first. It uses more GPU
memory and does more token work per call. If memory remains, increase
graph-microbatch-size to run more complete layer/head graphs in parallel. Gate
projection and scoring are batched across the graph microbatch. Only RMSNorm
is grouped by transformer layer.

Before a full run, process one context and then resume from the next one:

```bash
python -B train_graph.py \
  --model "$MODEL_ID" \
  --gate-checkpoint fastkvzip \
  --teacher-cache-dir "$TMPDIR/teacher-cache" \
  --output-dir graph_checkpoints/pilot \
  --max-contexts 1

python -B train_graph.py \
  --model "$MODEL_ID" \
  --output-dir graph_checkpoints/pilot \
  --resume graph_checkpoints/pilot/last.pt
```

A one-context pilot creates `last.pt`. By default it is saved once per
completed epoch; validation also runs once per completed epoch. `best.pt` is
written after an improved full validation sweep. Pass `--no-save-best` to keep
only the repeatedly replaced `last.pt`. Evaluate a completed checkpoint with:

```bash
python -B eval_graph.py \
  --graph-checkpoint ../graph_checkpoints/two-phase/best.pt \
  --run-dir ../results/experiment
```

The default evaluation task is `scbench_kv`.

#### RULER and complete benchmark coverage

Graph evaluation runs the complete prepared/filtered split by default. Use
`--num 1` for a pilot or `--idx`/`--num` for a range; limits count contexts,
not questions. SQuAD retains all questions for each selected context. The
shared `--data all` inventory contains **107** configurations: 27 SCBench
files (including supported length variants), SQuAD, GSM, and 78 RULER
task/length pairs. Names are never automatically shortened for a model.
Agentic remains available for training but is excluded from `all` and this grid.

```bash
python -B eval_graph.py \
  --graph-checkpoint ../graph_checkpoints/answer/q25a-s40n40-n200e2-uniform-s0/best.pt \
  --data ruler_4k \
  --level pair --window-size 0 \
  --ratios 0.75 0.50 0.40 0.30 0.20 --full-cache-answer \
  --existing-results resume --run-dir ../results/uniform-ruler-4k
```

`--data ruler` selects all thirteen tasks at 4K, 8K, 16K, 32K, 64K and
128K; `ruler_128k` selects that length; `ruler_niah_single_1_128k` selects
one task. Each pinned split has 500 examples. The requested task's Parquet file
is downloaded once from
[lighteval's Qwen2.5-Instruct collection](https://huggingface.co/datasets/lighteval/RULER-4096-Qwen2.5-Instruct)
using the normal Hugging Face Hub cache, then read through Datasets. Other tasks
are not downloaded. A warm cache can be reused in a new process without network
access. Keep `HF_HOME`/`HF_HUB_CACHE` and `HF_DATASETS_CACHE` in durable per-account
storage to reuse them across jobs; scratch does not survive a job. There is no
preparation step or data-dir option. Dataset commits are pinned in `data/ruler.py`
and saved in results.
Worked demonstrations stay in context; the final question and answer cue
stay outside compression. Multiple targets and QA aliases remain separate.

`--ruler-prompt-mode graphkv` (default) reuses the checkpoint's exact saved
prefix and normal query wrapper. `official` preserves the published task
wording without the added GraphKV general instruction or `Q:` prefix,
using the same model chat boundaries and generation path. It is a template
comparison, not byte-identical replay of upstream inference. Official-mode
task/result keys end in `_official`, and manifests prevent mixing modes.

Scoring follows [RULER's official metrics](https://github.com/NVIDIA/RULER/blob/c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a/scripts/eval/synthetic/constants.py):
case-insensitive substring recall for NIAH/VT/CWE/FWE and any-alias matching
for QA. Generation limits are respectively 128/30/120/50/32 tokens. The
task means in `ruler_macro_averages` are equal-weighted by task, separately
per length and ratio; only `complete: true` means all thirteen tasks and
their full splits are covered. Partial means are not complete benchmark scores.

Generation now removes a final token only if it is an effective EOS token;
length-capped answers retain their final real token. Old result directories
can still be parsed but cannot be resumed under this corrected generation
revision. Use a new directory. Old generated-answer cache entries become
misses rather than being overwritten or silently reused. `datasets.json`
records true full-split sizes before evaluation. Unknown sizes (Agentic and
bounded MRCR reads) are recorded as `null`: local scores remain available, but
these runs are never marked complete or uploaded as full benchmarks. MRCR's
default full evaluation exhausts its filtered split; limited reads do not scan
the rest just to determine a size. Interrupted runs and small pilots cannot be
misreported as complete benchmarks.

Dataset selectors do not take a model name or silently substitute SCBench
lengths. For legacy results, explicitly select the variant that was evaluated
(for example `--data scbench_prefix_suffix_short`).

#### Evaluation-only W&B destination and production grid

`--wandb-run-id NEW_ID` binds either graph evaluator to a new evaluation
run without modifying the checkpoint. Without the flag, the checkpoint's
destination remains the default. Binding does not enable uploads: GPU
workers omit `--log-to-wandb`, and pilots make no W&B writes.

The [approved plan](../docs/plans/ruler-evaluation.md) defines checkpoint
verification on all three accounts, pilot inspection, exact resource-manifest
approval, submission-order tiers, and the new evaluation-only run. The
offline `slurm/plan_ruler_evaluation.py --dry-run` helper reads a JSON spec
from stdin and prints the exact 107-job manifest. Its tested spec example
is in `tests/test_ruler_grid.py`; resource/time estimates must come from
measurements, not its fixture values. It uses the standard evaluation
wrapper, balances individual GPU slots, and refuses missing checkpoint
verification or unresolved training dependencies. It never submits jobs.

Priority tiers are `scbench_kv`, RULER 4K, RULER 8K, then the remaining
80 configurations. These are submission order, not completion dependencies;
Slurm can still start jobs out of order. Preserve every attempt in the
durable receipt and never retry an unknown submission outcome blindly.

`slurm/collect_ruler_evaluation.py GRID_DIR --approved-sha256 SHA --collect`
reuses the serial collector's existing receipts, local snapshots and upload
history. Its default is a read-only local dry run. For a redistributed retry
batch, retain the original `manifest.json`/`receipt.json` and add an explicitly
approved `retry-manifest.json`/`retry-receipt.json`, bound with
`--retry-approved-sha256 SHA`. It collects from each successful attempt's
approved account, waits for all possible writers to become terminal, and keeps
the original W&B destination and every failed attempt. Tests in
`tests/test_ruler_collection.py` show the receipt contract.

Only zero-output jobs move accounts; partial outputs resume in their original
directory on their original account. Recheck live jobs and output counts before
any cancellation or submission. Stage fixes separately from running/queued
jobs' checkouts; do not pull new code into their live runtime.

One serial coordinator can poll synchronized worker result directories and
upload complete benchmarks without waiting for the grid:

```bash
python -B -m results.coordinator ../results/worker-one ../results/worker-two \
  --wandb-run-id "$EVALUATION_RUN_ID" \
  --wandb-project graphkv-answer-qwen25-7b1m-s40n40-grid-v1 \
  --wandb-entity danielohayon2016-ben-gurion-university-of-the-negev
```

Each invocation is one pass; missing/partial workers are skipped. Repeated
passes reuse existing conflict checks and upload only missing metric points.
Run exactly one coordinator, outside GPU workers. It does not write worker
directories and prints combined coverage and RULER macro-averages as JSON.
Keep all workers on the same new evaluation destination; production results
must not be uploaded to the source training run.

Full-cache answer generation is enabled by default. Add
`--no-full-cache-answer` when another run already provides the same base-model
reference. The result then stores `"full__": null`; pruned answers and ground
truth are still stored.

Evaluation requires the pinned `datasets==4.0.0` to read current Hub metadata.
The protected local window is a hard minimum. If it is larger than a requested
retention budget, the saved actual ratio is higher than the request.
Whole-context W&B evaluation reports model selection outside that window
instead of total actual retention.
Pass `--window-size 0` to disable protection at every context length. Evaluation
manifests record `window_revision=1`; results from the old short-context fallback
remain readable but require a new directory for corrected evaluation.
Pass a nonzero integer such as `--window-size 4096` for the existing adaptive/fixed
policy. Pass a ratio such as `--window-size 0.02` to protect that fraction of
the context at every context length.

`--run-dir` is required. All graph evaluations use this one resumable result
layout. Add `--existing-results resume` to continue an existing run. Repeated
retention ratios are deduplicated before evaluation.

The checkpoint restores the model identifier, exact prefix tokens, prefill
chunk size, and token/graph microbatch settings.

Training has three teacher-data modes: online generation (the default), a full
cache with `--teacher-cache-dir`, and a scores-only cache with both
`--teacher-cache-dir` and `--teacher-cache-scores-only`. The scores-only flag
must be repeated for every run that reads those files: it stores scores and
token metadata but no activations, so training replays the teacher hidden
states for each context. Cache files are validated against the model and
prefill chunk and are never overwritten automatically. A complete full cache
unloads the base LLM after constructing the student; scores-only mode keeps it
resident for hidden-state replay. If a full-cache file later goes missing,
training reloads the LLM only when that file is needed. Hugging Face model
caches, graph checkpoints, and W&B logs still use disk.

W&B is online by default. It logs training losses, learning rates, the mean
layer/head alpha, fractional epoch, and cumulative scored training tokens under
`train/`. It logs mean validation BCE under `validation/`, and
forward/backward time per scored context token under `timing/`. The terminal
also shows one context-level training progress bar with the `train/` metrics.
Its system monitor supplies GPU metrics; the trainer does not emit a separate
`gpu/` metric section. Use `--wandb-mode offline` only when the run should not
sync immediately.

On Slurm, push the PR first, then run sres immediately before submission.
Prefer rtx_pro_6000:1; use rtx_6000:1 when it has better live availability.
The one-context pilot uses one GPU, one hour, --mem=60G, and --tmp=40G. Put
the teacher cache in node-local scratch and checkpoints/W&B logs in durable
shared storage. Recheck live limits before a later full cached run using
--tmp=600G.

### Answer-supervised Agentic training

Run these from `prefill/`. Answer training starts from a random graph with a
model, or from graph weights; a full resume restores optimizer, scheduler, and
cursor state:

```bash
python -B train_graph_answer.py \
  --model "$MODEL_ID" \
  --validation-retention-ratio 0.2 \
  --output-dir ../graph_checkpoints/answer/random

python -B train_graph_answer.py \
  --graph-checkpoint ../graph_checkpoints/graph/best.pt \
  --validation-retention-ratio 0.2 \
  --output-dir ../graph_checkpoints/answer/from-graph

python -B train_graph_answer.py \
  --resume ../graph_checkpoints/answer/from-graph/last.pt \
  --validation-retention-ratio 0.2 \
  --output-dir ../graph_checkpoints/answer/from-graph
```

On resume, use the validation retention ratio saved in the checkpoint (the
example's original value is `0.2`).

New runs shuffle individual training question-context pairs every epoch by
default. `--no-shuffle-data` preserves their dataset order. Each epoch's order
is generated by a separate `random.Random(seed + epoch)` instance, so it is
reproducible on resume and does not consume retention randomness. Shuffling
does not change the selected data or validation split, and does not guarantee
distinct contexts within each batch. `--train-context-count` counts deduplicated
question-context pairs, not unique documents.

Use `--gradient-accumulation-steps 8` for eight questions per optimizer update
(default: `1`). Questions are processed sequentially; only scorer parameter
gradients persist across questions. The update averages each question's mean
answer loss equally, regardless of answer length. An epoch's final smaller
batch uses its actual size: five training questions with accumulation `2`
produce batches of `2, 2, 1`. Learning rates are not automatically scaled.

`--train-context-count` selects the dataset reused across `--epochs` (before
any fallback validation holdout). Stage 2 has no separate `--max-contexts`
limit. For a smaller pilot, select fewer question-context pairs and epochs.
For example, `--train-context-count 12 --epochs 1` leaves ten training questions
with the fallback holdout; accumulation `8` produces updates of `8, 2`.

Checkpoints contain completed updates only. After an interruption, `--resume`
continues from the saved epoch, question offset, and optimizer state, preserving
the batch sequence. Work since the last saved checkpoint is repeated. Use
`--save-strategy steps --save-every N` to checkpoint every N optimizer updates;
the default saves after each epoch. `best.pt` is written when validation improves.

Both new controls are saved in the checkpoint and cannot change under
`--resume`. Old answer-training checkpoints resume with accumulation `1` and
shuffling disabled. Use `--graph-checkpoint` to start a new run from weights
with different settings; this uses the new-run defaults.

The default `linear` retention schedule starts at `--retention-max` and decays
to `--retention-min` over the global optimizer-step horizon across all epochs;
it does not reset at epoch boundaries and uses one ratio for the whole batch.
`uniform` instead samples once per training question between those bounds.
The update horizon is `epochs * ceil(training_questions / accumulation)`;
learning-rate schedulers advance after each update. `LinearWarmupCosineLR`
requires at least two total updates. Save/evaluation cadence with strategy
`steps` counts optimizer updates; strategy `epochs` counts completed epochs.

W&B emits one training row per optimizer update. `train/answer_nll` and
`train/answer_token_accuracy` are equal-question batch averages. Score-gradient
norm diagnostics are averaged across questions; gate/mixer gradient norms are
measured after averaging the accumulated parameter gradients. `train/examples`
counts processed questions, `train/optimizer_step` counts updates, and
`train/batch_examples` records the actual batch size. `train/epoch` remains
fractional data progress and `train/tokens` counts cumulative context tokens.
`optimizer_step` is the only training step counter. Question counts are derived
from the saved epoch and offset. Training and any due validation metrics are
logged together. Logging uses the standard W&B **Step** axis, explicitly set to
the completed optimizer count (the first update is Step 1). Each update commits
one row, including any due validation metrics. The previous custom-axis default
is cleared; existing manually configured panels can select **Step** in their
X-axis settings.
W&B configuration includes `prefill_chunk` alongside the graph and token
microbatch settings so that these performance settings can be compared between runs.
There is no separate saved global or W&B step. Validation NLL and accuracy
remain answer-token-weighted at the fixed validation retention ratio.

Agentic answers are resolved lazily after full prefill. Pass a durable
`--answer-cache-dir` to reuse completed answers; missing entries are filled as
they are needed. Without it, answers are generated for that step and discarded.
There is no separate cache-prewarm job.
A persistent answer cache requires a Hugging Face Hub model ID resolved to an
immutable commit. Local model directories are supported only without
`--answer-cache-dir`.

Evaluate the matching pair/head, unprotected-window protocol with the same
optional durable answer cache:

```bash
python -B eval_graph.py \
  --graph-checkpoint ../graph_checkpoints/answer/from-graph/best.pt \
  --data agentic \
  --level pair-head \
  --window-size 0 \
  --answer-cache-dir /durable/agentic-answer-cache \
  --run-dir ../results/agentic-from-graph
```

### Efficiency Measurement
You can measure the memory and decoding speed:
```python
python -B profiling.py -p $context_len -r $compression_ratio
```
- Set `-r 1.0` to profile a case using the full KV cache.
