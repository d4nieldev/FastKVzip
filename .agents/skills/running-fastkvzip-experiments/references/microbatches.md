# GPU and Microbatch Reference

Use these as starting points for Qwen3-8B GraphKV workloads. Confirm the
allocated GPU model and total memory with `nvidia-smi`; names alone are not
enough.

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
largest intended context.

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
