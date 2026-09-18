# Pilot summary — granola and GPS on GPU

> **Measured before the learned RNF projection.** These numbers come from
> `d9f6e82`, which concatenated the raw random features. The commit that added
> the `granola_rnf_mlp` and changed the default RNF width to `graph_dim // 4`
> lands after them, so GraNoLa's parameter counts here are slightly low and its
> memory slope is a slight over-estimate. The GPS figures are unaffected.

Qwen2.5-7B-Instruct-1M, PR #32 head `d9f6e82`, 81 jobs, all on `danieloh`.
No code was changed during the pilot. Detail in `results.md`, open items in
`findings.md`, chronology in `progress.md`.

## Does it work?

Yes. Every pipeline, for both new paths.

| Pipeline | granola | gps |
|---|---|---|
| Stage-1 training | ✅ | ✅ |
| Evaluation, up to all 112 graphs at once | ✅ | ✅ |
| Checkpoint save, reload, resume to epoch 2 | ✅ | ✅ |
| Stage-2 answer training | ✅ NLL 1.33 | ✅ NLL 1.39 |
| 48 GiB card | ✅ 39.0 GiB | ✅ 36.3 GiB |
| Hyperparameter variants | ✅ 6 of 6 | ✅ 9 of 9 |
| Refusals for settings it ignores | ✅ | ✅ |

**No defects found.** Every failure was either a deliberate OOM bracket, a
refusal that was supposed to fire, or a mistake in my own test setup.

## The three results worth acting on

### 1. GPS is the smaller model, not the larger one

At `--graph-dim 32`: gps 26.9M parameters against the implicit mixer's 39.3M.
The implicit mixer needs three hidden-width projections, GPS needs two, and the
GPS stack's own tensors are negligible at graph width.

This reverses the assumption behind the comparison design. A GPS win at equal
width cannot be explained by extra capacity. Worth re-deciding how to match the
two architectures before the real grid.

### 2. GraNoLa is materially cheaper in memory than batchnorm

Peak training memory is linear in graph microbatch, same intercept for every
arm, different slopes:

| Arm | GiB per graph-microbatch unit | vs batchnorm |
|---|---:|---:|
| granola | 1.18 | -26% |
| gps | 1.49 | -7% |
| batchnorm | 1.60 | — |

Shared intercept 19.5 to 19.6 GiB: weights, KV cache, resident teacher.

These are per unit of **graph microbatch at token 16000**. Divide by 8 for the
per-width-unit slopes in "Sizing a future grid" below, which is the form to use
when the token microbatch is not 16000.

GraNoLa normalizes at graph width 32 and projects out afterwards; batchnorm
normalizes at hidden width 3584. More parameters, far less activation memory.
It is the only arm that trains at width 448.

### 3. Every architecture knob is free except GraNoLa's width

Depth, attention heads, random features and the redraw schedule all cost under
0.1 GiB. **GPS at graph width 64 costs 0.1 GiB for 2.1x the parameters.** Only
GraNoLa's width costs anything (+6.2 GiB for 32 to 64).

So GPS capacity can be bought almost for free, and its knobs can be tuned on
quality alone.

## Sizing a future grid

```
peak GiB = 19.6 + slope x width
width    = (token_microbatch / subgraph_size) x graph_microbatch
```

| Arm | slope per width unit |
|---|---:|
| granola | 0.147 |
| gps | 0.187 |
| batchnorm | 0.200 |

**To pick settings:** card GiB, minus 19.6, divided by the slope, is the width
budget. Split it across the two axes however suits throughput -- the split does
not change memory.

The 19.6 GiB intercept is model weights, KV cache and the teacher held resident
by the scores-only cache, and is the same for all three arms. The slope is the
architecture.

Verified over three arms, four widths (56, 112, 224, 448), both axes and four
token microbatch settings. The slopes were fitted on the **graph** axis alone
and then predicted the **token** axis to 0.1 GiB at every point, and a
different GPU model to 0.1 GiB. Peak memory is a property of the tensors, not
the card.

| Card | gps | granola | batchnorm |
|---|---:|---:|---:|
| 95 GiB, stage-1 training | gm 48 | gm 63 | gm 46 |
| 44 GiB, evaluation | gm 32 measured | gm 40 measured | pilot from 16 |

**Stage 2 caps at width 224** for both architectures. Above it, one 11.96 GiB
allocation fails, and it follows width rather than either axis -- the inverted
split at the same width fails identically. Getting past it means lowering the
width, not rearranging the split.

Two further operational notes:

- **The first job of a grid is the binding constraint.** A cold scores cache
  adds the teacher scoring prefill, and that job peaked higher at gm16 than a
  warm job did at gm28.
- **`rtx_6000` covers more than one card.** Two jobs under the same request got
  44.4 and 47.4 GiB usable. Size against 44.

## What this pilot does NOT tell you

- **Nothing about quality.** Every evaluation returned 90.00 at every ratio:
  one example, mixers trained on two contexts. It proves the pipeline runs.
- **Nothing about speed.** Single runs on a shared cluster, one of which hit a
  failing node.
- The deferred Stage C comparison is still deferred, and finding 1 above
  changes how it should be set up.

## Two traps worth remembering

1. **Never read a resumed run's peak from W&B.** Resume restores the W&B run ID
   from the checkpoint, so a resumed run appends to its parent's history and
   the maximum over it is the parent's peak.
2. **A resume must reuse the checkpoint's microbatch settings.** They are saved
   invariants, refused at the CLI before any GPU work. Correct behaviour; it
   cost me three jobs to rediscover.
