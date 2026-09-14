# GPU and Microbatch Reference

Two workloads are covered: the original Qwen3-8B stage-1 training (next
section) and Qwen2.5-7B-Instruct-1M answer training with `train_graph_answer.py`
(measured 2026-09-13, see "Answer training" below). Confirm the allocated GPU
model and total memory with `nvidia-smi`; names alone are not enough.

## What predicts memory: scorer width

Neither microbatch flag predicts peak memory alone. Their product does:

```
scorer width = (token_microbatch_size / subgraph_size) x graph_microbatch_size
```

The scorer treats each (layer, KV head) pair as one graph; Qwen2.5-7B-1M has
28 x 4 = 112. With `--subgraph-size 2000`, `--token-microbatch-size` packs
`token / 2000` subgraphs into one call, so one call holds `subgraphs x graphs`
Gram matrices. Three different splits at width 56 all peaked at exactly
40.9 GiB. Width predicts memory; the split does not. Speed differences between
splits at fixed width were inside single-sample noise (2% on the 96 GiB card).

## Answer training (Qwen2.5-7B-Instruct-1M, `--subgraph-size 2000`)

| Allocated GPU | Graph microbatch | Token microbatch | Prefill chunk | Width | Peak | Notes |
|---|---:|---:|---:|---:|---|---|
| RTX 6000 Ada, 48 GB (47.4 usable) | 14 | 8,000 | 16,000 | 56 | 40.9 GiB | Ceiling; width 84 OOMs. Measured on the 117K-token worst case. |
| RTX PRO 6000, 96 GB | 112 | 4,000 | 131,072 | 224 | 86/96 GiB | 1.33x faster than 28/16,000 (222 vs 295 s/step), memory flat from step 10. |
| RTX PRO 6000, 96 GB | 56 | 8,000 | 131,072 | 224 | 86/96 GiB | Same speed as 112/4,000 within 2%. |
| RTX PRO 6000, 96 GB | 28 | 16,000 | 16,000 | 224 | 51/96 GiB | Old default; leaves 45 GiB idle and is the slowest. |

`--prefill-chunk`: on the 48 GB card, 8,000 to 16,000 was 29% faster and free;
16,000 to 131,072 cost 11.5 GiB and bought nothing. All three 96 GB rows share
width 224, so their 51 to 86 GiB gap is prefill, not the microbatch split.
Re-measure the 96 GB card at prefill 16,000 before assuming 89% memory is
needed for the 1.33x speedup.

Headroom at 86/96 GiB is about 10 GiB; an unusually long context can still
OOM. Every s/step figure is a single update gap on a shared node; re-measure
over several updates when the choice matters. Source: `AGENTS.md` at the repo
root and W&B project `graphkv-answer-qwen25-7b1m-s40n40-grid-v1`.

## Qwen3-8B stage-1 training

## Training defaults

| Allocated GPU | Token microbatch | Full-context graph microbatch | 2K-chunked graph microbatch |
|---|---:|---:|---:|
| RTX 6000, about 48 GB | 16,000 | 8 | 8 |
| RTX PRO 6000, about 96 GB | 16,000 | 16 | 24 |

For RTX 6000 chunked training, 12 is a candidate only after a representative
pilot. Fall back to 8 without fresh evidence.

These choices reflect completed runs: full-context graph microbatch 16 OOMed
on RTX 6000 while 8 completed; chunked graph microbatch 8 completed with more
headroom. RTX PRO 6000 completed full-context 16 at roughly 82% peak memory and
chunked 24 at roughly 68% in the validated pilots, including scores-only cache
operation.

The RTX 6000 defaults assume the comparable completed cache behavior. A
scores-only cache keeps the teacher resident, so do not assume those values fit
on a 48 GB card; prefer RTX PRO 6000 or pilot below the table and promote from
observed memory.

`--token-microbatch-size` bounds token-width work. `--graph-microbatch-size`
sets how many complete layer/head graphs run together. They are independent,
and their interaction controls peak memory. For 2K subgraphs with token
microbatch 16,000, `--subgraphs-per-step 8` covers one token microbatch per
optimizer step.

## Evaluation defaults

Start evaluation conservatively:

| Allocated GPU | Token microbatch | Graph microbatch |
|---|---:|---:|
| RTX 6000, about 48 GB | 16,000 | 8 |
| RTX PRO 6000, about 96 GB | 16,000 | 16 |

Evaluation has no backward pass, but `full` token batching combined with `all`
graph batching can still OOM. Increase only after an evaluation pilot on the
largest intended context. For Qwen2.5-7B-1M answer checkpoints the training
widths above are safe lower bounds (evaluation used 16,000 / 8 on RTX PRO 6000
for LongBench v2 at 998K tokens, 76 GiB peak); pilot upward from them rather
than assuming the training ceiling.

## Promotion procedure

Before increasing either value:

1. Inspect peak GPU memory from recent completed W&B runs with the same model,
   training mode, graph construction, dtype, and context distribution.
2. Pilot at least one ordinary 10K–30K context and one representative
   concatenated context of at least 100K tokens.
3. Use the largest value whose worst observed peak leaves about 15% headroom.
4. Record GPU model/memory, both microbatches, longest context, peak, and run ID
   in the grid receipt.

If the pilot OOMs, reduce graph microbatch first unless evidence points to the
token-width phase. Do not extrapolate linearly across GPU models.

## Resume constraint

The current training checkpoint validates token and graph microbatch settings
as configuration invariants. Resume a saved checkpoint with the same values.
If an OOM requires smaller values, use a larger compatible GPU without changing
them, or obtain approval to restart the logical run from initial weights while
preserving its W&B and dashboard identity. Never edit checkpoint metadata.
