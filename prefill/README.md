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

The default is two-phase training with the released FastKVzip gate:

```bash
python -B train_graph.py \
  --model "$MODEL_ID" \
  --gate-checkpoint fastkvzip \
  --output-dir graph_checkpoints/two-phase \
  --wandb-mode offline
```

Use `--training-mode joint` for one joint gate/graph update per context:

```bash
python -B train_graph.py \
  --model "$MODEL_ID" \
  --gate-checkpoint fastkvzip \
  --training-mode joint \
  --output-dir graph_checkpoints/joint \
  --wandb-mode offline
```

Before a full run, process one context and then resume from the next one:

```bash
python -B train_graph.py \
  --model "$MODEL_ID" \
  --gate-checkpoint fastkvzip \
  --output-dir graph_checkpoints/pilot \
  --max-contexts 1 \
  --wandb-mode offline

python -B train_graph.py \
  --model "$MODEL_ID" \
  --output-dir graph_checkpoints/pilot \
  --resume graph_checkpoints/pilot/last.pt \
  --max-contexts 1 \
  --wandb-mode offline
```

A one-context pilot initially creates only `last.pt`; `best.pt` is written after validation. Evaluate a completed checkpoint with:

```bash
python -B eval_graph.py \
  --graph-checkpoint graph_checkpoints/two-phase/best.pt \
  --data squad \
  --tag experiment
```

The checkpoint restores the model identifier, exact prefix tokens, prefill chunk size, and token/graph microbatch settings. Evaluation result tags are always namespaced as `_graph` or `_graph_<tag>` so they do not overwrite baseline results.

Training generates teacher activations and scores online; hidden states, scores, and graph edges are not written as dataset artifacts. Hugging Face model caches, graph checkpoints, and offline W&B logs still use disk. FAISS neighbor search runs on CPU through `faiss-cpu`; use the pilot to measure CPU RAM, GPU allocated/reserved memory, and GPU utilization before choosing a full job size.

On Slurm, start with the one-context command on one GPU. Choose the partition, time limit, CPU memory, and GPU type from the cluster's current site policy and availability, then size the full job from the pilot measurements rather than copying a fixed resource request.

### Efficiency Measurement
You can measure the memory and decoding speed:
```python
python -B profiling.py -p $context_len -r $compression_ratio
```
- Set `-r 1.0` to profile a case using the full KV cache.
