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

For the cluster setup, pilot workflow, reusable `sbatch` scripts, and experiment controls, see [the experiment guide](../docs/graph-fastkvzip-experiments.md).

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
    X' = X + alpha * LeakyReLU(Normalize(Y1 S W))

Every layer/KV head has independent base mixer weights. It never materializes
a token-by-token adjacency matrix. `--normalization` selects `none`,
`batchnorm` (the default), or `granola`; `--normalization-sharing` selects
learned normalization parameters per `graph`, `layer`, or `global`. GraNoLa also
accepts `--granola-gnn-depth`, `--granola-mlp-depth`, and
`--granola-rnf-dim`. Other mixer controls are graph-dim (default 32),
gram-normalization, leaky-relu-slope, alpha-init, graph-microbatch-size, and
token-microbatch-size. Checkpoint/validation controls are save-strategy,
save-every, save-best, eval-strategy, and eval-every.

The GraNoLa option is the scalable signed weighted-sum GIN adaptation used by
this implicit low-rank graph; it is not the [DEAR reference implementation's](https://github.com/HekpoMaH/DEAR/blob/master/models/gnns.py#L127)
dense max-aggregation MPNN and does not claim the [paper's](https://arxiv.org/abs/2404.13344)
full universality result.

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
  --run-dir ../results/experiment \
  --data squad
```

Full-cache answer generation is enabled by default. Add
`--no-full-cache-answer` when another run already provides the same base-model
reference. The result then stores `"full__": null`; pruned answers and ground
truth are still stored.

Evaluation requires the pinned `datasets==4.0.0` to read current Hub metadata.
The protected local window is a hard minimum. If it is larger than a requested
retention budget, the saved actual ratio is higher than the request.

`--run-dir` is required. All graph evaluations use this one resumable result
layout. Add `--existing-results resume` to continue an existing run. Repeated
retention ratios are deduplicated before evaluation.

The checkpoint restores the model identifier, exact prefix tokens, prefill
chunk size, and token/graph microbatch settings.

Training generates teacher activations and scores online by default. With a
teacher-cache directory, each encountered training/validation context is
written atomically once and reused in later epochs or resumes. Cache files are
validated against the model and prefill chunk and are never overwritten
automatically. When the full expected cache is present, training unloads the
base LLM after constructing the student. If a file later goes missing, it
reloads the LLM only when that file is needed. Hugging Face model caches, graph
checkpoints, and W&B logs still use disk.

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

### Efficiency Measurement
You can measure the memory and decoding speed:
```python
python -B profiling.py -p $context_len -r $compression_ratio
```
- Set `-r 1.0` to profile a case using the full KV cache.
