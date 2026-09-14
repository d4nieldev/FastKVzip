# FastKVzip — agent notes

## Microbatch tuning for answer training

`Qwen/Qwen2.5-7B-Instruct-1M`, `train_graph_answer.py`, `--subgraph-size 2000`.

### The number that predicts memory: scorer width

Neither microbatch flag predicts memory on its own. Their **product** does:

```
scorer width = (token_microbatch_size / subgraph_size) × graph_microbatch_size
                └── subgraphs packed per call ──┘        └── graphs per call ──┘
```

The scorer treats each (layer, KV head) pair as an independent graph — 28 × 4 =
**112 graphs** for this model, so `--graph-microbatch-size 112` means all of
them at once. Since the subgraph-batching merge, `--token-microbatch-size` packs
`token / subgraph_size` subgraphs into one call. One call therefore holds
`subgraphs × graphs` instances, each with a 2000×2000 Gram matrix.

Verified on RTX 6000 Ada: three different flag pairs at width 56 all peaked at
**exactly 40.9 GiB**. Width predicts memory; the split does not change it.

### RTX 6000 Ada, 48 GiB (47.37 usable) — measured 2026-09-13

KL-loss answer training, worst-case 117,373-token contexts. Numbers preserved
here; the source W&B project `graphkv-answer-kl-pilot` has been deleted.

| graph | token | prefill | width | peak | s/update |
|---|---:|---:|---:|---|---:|
| 16 | 2000 | 8000 | 16 | 32.0 GiB | 204.5 |
| 16 | 2000 | 16000 | 16 | 31.8 GiB | 145.1 |
| 16 | 2000 | 131072 | 16 | 43.3 GiB | 146.0 |
| 14 | 8000 | 16000 | 56 | 40.9 GiB | **126.0** |
| 28 | 4000 | 16000 | 56 | 40.9 GiB | 196.6 |
| 56 | 2000 | 16000 | 56 | 40.9 GiB | 130.3 |
| 84 | 2000 | 16000 | 84 | — | OOM |
| 112 | 2000 | 16000 | 112 | — | OOM |
| 112 | 4000 | 8000 / 16000 | 224 | — | OOM |

**Use graph 14 / token 8000 / prefill 16000 on this card.** Width 56 is the
ceiling; 84 OOMs.

Note these are worst-case contexts. In a real 200-context Agentic run the
average is ~54k tokens and the observed rate was ~48 s/update, not 126.

### `--prefill-chunk`: 16000 is optimal, bigger is waste

At fixed scorer width, going 8000 → 16000 is **29% faster and free**
(32.0 → 31.8 GiB). Going 16000 → 131072 costs **+11.5 GiB and buys nothing**
(146.0 vs 145.1 s/update — identical).

This corrects the earlier version of this file, which recommended
`--prefill-chunk 131072`. That came from misreading the 96 GiB table below: all
three of its rows have **the same width, 224**, so the 51 vs 86 GiB gap was
caused by prefill alone, not by trading graph width for token width. Prefill was
buying memory, not speed.

### RTX PRO 6000, 96 GiB — measured 2026-09-13

| flags | s/step | width | GPU mem |
|---|---|---|---|
| `--graph-microbatch-size 28 --token-microbatch-size 16000` | 295 | 224 | 51/96 GiB (53%) |
| `--graph-microbatch-size 112 --token-microbatch-size 4000 --prefill-chunk 131072` | 222 | 224 | 86/96 GiB (89%) |
| `--graph-microbatch-size 56 --token-microbatch-size 8000 --prefill-chunk 131072` | 228 | 224 | 86/96 GiB (89%) |

All three are width 224, which is why their speeds are close. The 51 → 86 GiB
jump is prefill 16000 → 131072. Given the Ada result that prefill above 16000
buys no speed, **re-measure this card at prefill 16000 before assuming 89% is
required** — width 224 at prefill 16000 may reach the same speed near 51 GiB.

### Speed caveat

Every s/update figure above is a **single** update-to-update gap on a shared
node. The 196.6 outlier at width 56 is out of line with its two neighbours
(126.0 and 130.3) and is more likely contention than a real effect of the split.
Do not treat small speed differences here as established; re-measure over
several updates if the choice matters.

### Do not extrapolate across cards

Two claims transferred from the 96 GiB card and were wrong on the 48 GiB card:

- "scorer width is nearly free" — width 224 OOMs on Ada at any prefill.
- "use prefill 131072" — costs 11.5 GiB for zero speed.

Pilot on the target GPU. A 2-update pilot takes ~6 minutes, and
`--train-context-start 0 --train-context-count 3` hits the 117,373-token worst
case immediately. Use `--train-context-count 3`, not 2: `LinearWarmupCosineLR`
requires at least two optimizer updates.

## Slurm: `SLURM_TMPDIR` is not set on `cs-*` / `ise-6000-*` nodes

`slurm/train_graph_answer.sbatch` exits with `SLURM_TMPDIR is required` on these
nodes. The prolog prints `SLURM_SCRATCH_DIR:` as **banner text only** — it is
never exported — while actually creating `/scratch/$USER/$SLURM_JOB_ID`.

Workaround, inserted before the existing check:

```bash
if [[ -z "${SLURM_TMPDIR:-}" && -d "/scratch/$USER/${SLURM_JOB_ID:-}" ]]; then
    SLURM_TMPDIR="/scratch/$USER/$SLURM_JOB_ID"
fi
```

Not yet upstreamed. Only the `ise-6000p-*` (PRO) nodes export `SLURM_TMPDIR`,
which is why the original grids never hit this.

## Stale reference

`.claude/skills/running-fastkvzip-experiments/references/microbatches.md` lists
graph 16 / token 16000 for RTX PRO 6000 and graph 8 for RTX 6000, written for
Qwen3-8B and for the pre-subgraph-batching meaning of `--token-microbatch-size`.
It has not been updated. Prefer the measurements above for the Qwen2.5-7B-1M
answer workload.
