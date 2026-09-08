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
one task. Each pinned split has 500 examples. Published data is streamed
from [lighteval's Qwen2.5-Instruct collection](https://huggingface.co/datasets/lighteval/RULER-4096-Qwen2.5-Instruct)
using normal Hugging Face caching; there is no preparation step or data-dir
option. Dataset commits are pinned in `data/ruler.py` and saved in results.
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
records true full-split sizes before evaluation, so interrupted runs and
small pilots cannot be misreported as complete benchmarks.

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

For a one-context pilot, add `--max-contexts 1`. With the default epoch
cadence, that pilot stops before validation and writes only `last.pt`; resume
from `last.pt`. `best.pt` appears only after a completed epoch runs validation.

The default `linear` retention schedule starts at `--retention-max` and decays
to `--retention-min` over the global optimizer-step horizon across all epochs;
it does not reset at epoch boundaries. `uniform` instead samples once per
training example between those bounds.

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
