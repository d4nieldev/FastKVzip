# Pilot measurements

> **Measured before the learned RNF projection.** These numbers come from
> `d9f6e82`, which concatenated the raw random features. The commit that added
> the `granola_rnf_mlp` and changed the default RNF width to `graph_dim // 4`
> lands after them, so GraNoLa's parameter counts here are slightly low and its
> memory slope is a slight over-estimate. The GPS figures are unaffected.

## The sizing formula

```
peak GiB = 19.6 + slope x width
width    = (token_microbatch / subgraph_size) x graph_microbatch
```

| Arm | slope per width unit |
|---|---:|
| granola | 0.147 |
| gps | 0.187 |
| batchnorm | 0.200 |

To size a run: card GiB, minus 19.6, divided by slope, is the width budget.
Split it across the two axes however suits throughput; the split does not
change memory.

### Verified points (phase 2 and phase 10)

| width | granola | gps | batchnorm |
|---:|---:|---:|---:|
| 56 | 27.8 | 30.0 | — |
| 112 | 36.0 | 40.5 | — |
| 224 | 52.4-52.5 | 61.4-61.5 | 64.4-64.5 |
| 448 | 87.5 | OOM | OOM |

Three arms, four widths, both axes, four token microbatch settings (4000,
8000, 16000 at fixed graph microbatch, and iso-width splits at each). The
slopes were fitted on the **graph** axis alone and then predicted the **token**
axis to within 0.1 GiB at every point, plus a different GPU model to 0.1 GiB.

The 19.6 GiB intercept is model weights, KV cache and the teacher held resident
by the scores-only cache. It is the same for all three arms. The slope is the
architecture.

## Read this first

**Trust levels differ by source.**

| Source | What it is | Trust |
|---|---|---|
| OOM / completed | Slurm outcome | ground truth |
| Evaluation peaks | `torch.cuda.max_memory_allocated`, printed by `eval_graph.py` | exact |
| Training peaks | W&B `system.gpu.0.memoryAllocatedBytes`, process memory including the caching allocator's reserve | indicative, not exact |
| Wall times | single runs on a shared cluster | not evidence |
| Benchmark scores | one example, mixers trained on two contexts | not evidence |

The training peaks are not monotonic in graph microbatch for the batchnorm arm
(see "Anomaly" below), which is why they are marked indicative. Every ceiling
claim rests on the Slurm outcome, not on those numbers.

## Stage-2 answer training

`train_graph_answer.py`, initialised from each arm's stage-1 checkpoint,
12 contexts, one epoch, subgraph 2000, graph microbatch 28 (width 224).

| Arm | peak GiB | % of 95 | wall | answer NLL | token accuracy |
|---|---:|---:|---:|---:|---:|
| batchnorm | 82.2 | 86 | 20m36s | — | — |
| gps | 68.0 | 71 | 9m23s | 1.391 | 0.680 |
| granola | 64.3 | 67 | 16m00s | 1.328 | 0.680 |

Both new paths train answer-supervised end to end. Stage 2 costs 12 to 18 GiB
more than stage 1 at the same width, and the arm ordering is the same in both.
Batchnorm at 86% leaves little room for a stage-2 grid.

### Stage-2 ceiling: width 448 fails for both, on the same allocation

| Arm | width 224 | width 448 |
|---|---:|---|
| gps | 68.0 GiB | OOM, 11.96 GiB requested, 9.93 free |
| granola | 64.3 GiB | OOM, 11.96 GiB requested, 2.55 free |

Both die in the backward pass asking for **the same 11.96 GiB block**. So the
stage-2 ceiling is not set by gradual growth, where the arms differ, but by one
large contiguous allocation that neither arm can satisfy at width 448. GraNoLa's
shallower slope does not help here: it got closer (2.55 GiB free against 9.93)
and still failed on the same request.

### Tested: the blocking allocation follows width, not tokens

The hypothesis above -- that a smaller token microbatch would get past it --
was **wrong**. Phase 11 ran the same width 448 with the split inverted:

| Arm | split | width | allocation | result |
|---|---|---:|---:|---|
| gps | tok 16000 / gm 56 | 448 | 11.96 GiB | OOM |
| gps | tok 8000 / gm 112 | 448 | 11.96 GiB | OOM |
| granola | tok 16000 / gm 56 | 448 | 11.96 GiB | OOM |
| granola | tok 8000 / gm 112 | 448 | 11.96 GiB | OOM |

The identical request appears at identical width under every split, on both
architectures. Neither axis helps.

Practical consequence: **stage 2 runs at width 224**, for either architecture.
Getting above it means lowering the width itself, not rearranging the split.
The width law governs the single largest allocation, not only the total.

## Anomaly RESOLVED — two measurement artifacts, no code involvement

A clean repeat of batchnorm at gm16 gives **45.2 GiB**, not 77.9. With that,
the warm training series is exactly linear:

| Arm | gm 16 | gm 20 | gm 28 | slope GiB/unit | intercept |
|---|---:|---:|---:|---:|---:|
| batchnorm | 45.2 | 51.6 | 64.4 | 1.60 | 19.6 |
| gps | 43.5 | — | 61.4 | 1.49 | 19.6 |
| granola graph | 38.3 | — | 52.4 | 1.18 | 19.5 |
| granola token | 38.3 | — | 52.4 | 1.18 | 19.5 |
| implicit none | 38.3 | — | — | — | — |

All three arms share an intercept of 19.5 to 19.6 GiB. That is the fixed cost:
model weights, KV cache, and the teacher kept resident by the scores-only
cache. Everything above it is the scorer, and the slope is what separates the
architectures.

**Correction:** the 38.3 GiB reading was first dismissed as coming from the
failing node. It was not. The requeued runs on healthy nodes reproduce 38.3
exactly, with 31 and 43 samples. The dead node cost those runs their wall time,
not their measurement.

**Note on the missing granola-graph row:** its own W&B run no longer appears
under that name. The phase-5 resume test copied its run directory including
`wandb_run_id.txt`, so the resumed job adopted and renamed the same W&B run.
That is the inheritance problem in the section below, seen from the other side.
Its gm16 value is taken from granola-token, which matched it exactly at gm28.

gm20 predicted from the batchnorm line is 51.6; measured 51.6. The GPS repeat
reproduced its original 43.5 exactly, and the gm28 repeat reproduced 64.4.

### Where 77.9 came from

Two separate artifacts, both mine, neither in the code:

1. **p0 ran with a cold cache.** It performed the extra teacher scoring prefill
   that every later run skipped, and PyTorch's allocator kept that transient
   reserved for the rest of the job. Process memory stayed high.
2. **The resume runs inherited their parent's W&B run.** Resuming restores the
   W&B run ID from the checkpoint, and my phase-5 test copied the whole run
   directory including `wandb_run_id.txt`. So `p5-resume-batchnorm` appended to
   p0's W&B history, and the maximum over that history is p0's cold-cache peak,
   not its own. It reported 77.9 because p0 did.

The same explains the other two resumes: `p5-resume-gps` read 44.0 against its
parent's 43.5, and `p5-resume-granola` read 38.9 -- each the maximum over a
shared history rather than its own peak.

### Corrected conclusions

- Training peaks **are** monotonic and linear in graph microbatch, with a
  shared intercept near 19.6 GiB (model weights, KV cache, resident teacher).
- Slope ranks the arms the same way the ceilings do: granola 1.13, gps 1.49,
  batchnorm 1.60 GiB per unit of graph microbatch.
- The earlier claim that "a cold first job peaks higher than a warm job at
  double the width" still holds -- 77.9 cold at gm16 against 64.4 warm at gm28
  -- but it is a cold-versus-warm statement, not evidence of non-monotonicity.
- **Never read a resumed run's peak from W&B**, in this pilot or a future grid.
  It shares its parent's history.

## Superseded: the original anomaly write-up

| Run | gm | peak GiB | cache |
|---|---:|---:|---|
| p0-implicit-batchnorm-gm16 | 16 | 77.9 | cold |
| p5-resume-batchnorm | 16 | 77.9 | warm |
| p2-implicit-batchnorm-gm28 | 28 | 64.4 | warm |

More work at a lower peak is backwards, and the two gm16 readings agree exactly
across different cache states, so it is reproducible rather than noise.

Most likely cause: the W&B metric is **process** GPU memory, which includes
PyTorch's caching allocator reserve, not `max_memory_allocated`. Reserved
memory is history dependent. The evaluation ladder, which reads torch's true
allocated peak, was perfectly linear and predicted a different GPU model to
within 0.1 GiB -- consistent with the metric being the problem, not the code.

Phase 8 (21450493-96) repeats batchnorm at gm 16, 20 and 28 with a GPS control
to confirm. **Not recorded as a defect in the implementation**; nothing about
it points at granola, GPS, or anything PR #32 introduced.


Qwen2.5-7B-Instruct-1M, 112 layer/head graphs, `--graph-dim 32`,
`--subgraph-size 2000`, `--subgraphs-per-step 8`, token microbatch 16000,
scores-only cache, `rtx_pro_6000` (94.97 GiB usable).

Width = (token microbatch / 2000) x graph microbatch.

Peak memory is the maximum of `system.gpu.0.memoryAllocatedBytes` sampled by
the W&B agent roughly every 15 seconds. **Read the sample count**: a peak from
very few samples may have missed the true maximum.

## Why GPS is smaller, even though it contains the implicit mixer

GPS does include the implicit message passing: `_GPSBlock.local` is a
`_PerGraphImplicitBranch`, the same tied-projection Gram aggregation. The
reason it is still the smaller model is *where* that aggregation runs.

Per graph, `--graph-dim 32`, hidden 3584 (from `param_breakdown.py`):

| Module | per graph | share |
|---|---:|---:|
| implicit `in_proj` (3584 -> 64, W_a and W_v fused) | 229,376 | 65% |
| implicit `out_proj` (32 -> 3584) | 114,688 | 33% |
| implicit batchnorm affine, hidden width | 7,169 | 2% |
| **implicit total** | **351,233** | |
| gps `in_proj` (3584 -> 32) | 114,688 | 48% |
| gps `out_proj` (32 -> 3584) | 114,688 | 48% |
| gps attention qkv | 3,072 | 1.3% |
| **gps `local.proj` -- the implicit message passing** | **2,048** | **0.9%** |
| gps ffn in + out | 4,096 | 1.7% |
| gps attention out, three layer norms | 1,216 | 0.5% |
| **gps total** | **239,809** | |

The implicit mixer reads from *and* writes back to hidden width, so it needs
three 3584-wide projections. GPS needs two, one in and one out, and does
everything else inside graph space. Its copy of the message passing is a 32->64
projection rather than a 3584->64 one: 2,048 parameters against 229,376.

The arithmetic closes exactly:

```
implicit (none)           38,535,280
  - one hidden projection -12,845,056
  + one GPS block          +1,168,384
                        = 26,858,608   <- measured
```

1,168,384 is also the measured per-depth increment, so GPS would need about
depth 11 to reach the implicit mixer's parameter count.

## Why GraNoLa is cheaper despite more parameters

From `ImplicitGraphMixer.normalized`, the branches receive different tensors:

- batchnorm: aggregate -> `out_proj` to 3584 -> **normalize at 3584** -> leaky ReLU
- granola: aggregate -> GNN at 32 -> **affine at 32** -> `out_proj` -> leaky ReLU

So batchnorm creates and saves a `[G, T, 3584]` intermediate for the backward
pass that granola never creates. At gm 28, T 2000, bf16 that is about 400 MB
per subgraph batch, roughly 3.2 GB across the eight subgraphs in a token
microbatch.

GraNoLa's extra parameters are real but all sit at graph width: 6,400 per
graph, with `[G, T, 32]` activations at about 3.6 MB. It takes roughly a
hundred of those to equal one hidden-width intermediate. The trade is 716,800
extra parameters to avoid a 112x wider activation.

### GraNoLa token adaptivity

| | gm 16 | gm 28 | gm 56 |
|---|---:|---:|---|
| granola graph | 38.3 | 52.4 | 87.5, completes |
| granola token | 38.3 | 52.4 | 88.6, **OOM** |

Free below the ceiling, +1.1 GiB at width 448 -- enough to tip it over.
Runtime was identical at gm 28 (2m45s both). Per-token gamma and beta are
`[G, T, 32]` rather than `[G, 32]`: small, but it scales with tokens.

### GraNoLa's width is graph_dim

The GNN runs entirely at `graph_dim`, 32 by default, and the RNF width defaults
to `graph_dim` as well, so block 0 takes `[R || rho]` at 64 and later blocks at
32. Nothing it touches is hidden width. That is why `--graph-dim 64` was the
only knob in the sweep that cost memory.

## Parameter counts

| Mixer | Parameters | Per graph | vs implicit-none |
|---|---:|---:|---:|
| implicit, `none` | 38,535,280 | 344,065 | — |
| implicit, `granola` | 39,252,080 | 350,465 | +716,800 (graph width) |
| implicit, `batchnorm` | 39,338,096 | 351,233 | +802,816 (= 2 x 3584 x 112) |
| **gps** | **26,858,608** | **239,809** | **-11,676,672** |

GPS is 68% of the implicit mixer at equal width. The implicit mixer needs three
hidden-width projections, GPS needs two.

## Training memory ladder (warm scores cache)

| Arm | gm | width | peak GiB | % of 95 | result | samples |
|---|---:|---:|---:|---:|---|---:|
| batchnorm | 16 | 128 | 77.9 | 81 | completed (**cold cache**) | 102 |
| batchnorm | 28 | 224 | 64.4 | 67 | completed | 20 |
| batchnorm | 56 | 448 | — | — | **OOM** | 0 |
| batchnorm | 112 | 896 | — | — | **OOM** | 0 |
| granola graph | 28 | 224 | 52.4 | 55 | completed | 20 |
| granola graph | 56 | 448 | 87.5 | 92 | completed | 32 |
| granola graph | 112 | 896 | — | — | **OOM** | 0 |
| granola token | 28 | 224 | 52.4 | 55 | completed | 20 |
| granola token | 56 | 448 | 88.6 | 93 | **OOM** | 8 |
| gps | 16 | 128 | 43.5 | 45 | completed | 20 |
| gps | 28 | 224 | 61.4 | 64 | completed | 9 |
| gps | 56 | 448 | — | — | **OOM** | 0 |
| gps | 112 | 896 | — | — | **OOM** | 2 |

## Evaluation memory ladder (exact peaks)

`eval_graph.py` records its own peak and prints it, so unlike training these
are exact rather than sampled. Checkpoints trained at gm 28, evaluated at the
graph microbatch shown, token microbatch 16000, `scbench_kv_short`, one example.

| Arm | gm 16 | gm 28 | gm 56 | gm 112 |
|---|---:|---:|---:|---:|
| gps | 26.1 | 33.8 | 51.7 | **87.7** |
| granola graph | — | 32.1 | 48.3 | **80.8** |
| granola token | — | 32.1 | — | — |
| implicit none | 27.9 | — | — | — |
| batchnorm | 31.3 | — | 70.0 | **OOM** |

Every arm but batchnorm evaluates with every graph resident at once.

### Fitted lines

Both ladders are linear in graph microbatch with the same intercept:

- gps: 0.64 GiB per unit, intercept 15.9 GiB
- granola graph: 0.58 GiB per unit, intercept 15.9 GiB

The shared 15.9 GiB is the model weights and KV cache; the slope is the scorer.
GraNoLa is about 10% cheaper per unit than GPS here.

### What fits a 48 GiB rtx_6000 — predicted, then measured

| Arm | gm | predicted | **measured** | card | headroom |
|---|---:|---:|---:|---|---:|
| gps | 32 | 36.4 | **36.3** | 44.4 GiB | 82% used |
| granola graph | 40 | 39.1 | **39.0** | 47.4 GiB | 82% used |

Both predictions landed within 0.1 GiB. Peak allocated memory is a property of
the tensors, not the card, so a line fitted on the 95 GiB card predicts the
smaller one directly. Only whether it *fits* changes.

Both evaluations produced the same scores as on the larger card.

**Watch the card name.** The two jobs were allocated 44.4 GiB and 47.4 GiB of
usable memory under the same `rtx_6000` request, so that gres name covers more
than one variant. Size against 44 GiB, not 48.

### Scores are not a quality signal here

Every evaluation returned score 90.00 with identical selection rates at every
ratio. That is one example against mixers trained for two contexts, so it says
the pipeline runs end to end and nothing about which architecture is better.
Do not read it as a comparison.

## Hyperparameter sweep: the knobs are nearly free

All 15 variants completed, all at graph microbatch 28 (width 224), same data.

| Variant | peak GiB | vs its baseline | parameters |
|---|---:|---:|---:|
| gps baseline (depth 1) | 61.4 | — | 26,858,608 |
| gps depth 2 | 61.5 | +0.1 | 28,026,992 |
| gps depth 3 | 61.5 | +0.1 | 29,195,376 |
| gps 2 attention heads | 61.4 | 0.0 | 26,858,608 |
| gps 8 attention heads | 61.4 | 0.0 | 26,858,608 |
| gps 16 random features | 61.4 | 0.0 | — |
| gps 64 random features | 61.4 | 0.0 | — |
| gps redraw every step | 61.4 | 0.0 | — |
| gps redraw never | 61.4 | 0.0 | — |
| **gps graph-dim 64** | **61.5** | **+0.1** | **56,010,864** |
| granola baseline | 52.4 | — | 39,252,080 |
| granola sharing layer | 52.4 | 0.0 | — |
| granola sharing global | 52.4 | 0.0 | — |
| granola gnn depth 2 | 52.4 | 0.0 | — |
| granola mlp depth 2 | 52.4 | 0.0 | — |
| granola rnf 16 | 52.4 | 0.0 | — |
| **granola graph-dim 64** | **58.6** | **+6.2** | — |

### What this means for tuning

1. **Every GPS knob is free.** Depth, heads, random features and the redraw
   schedule all leave peak memory unchanged within 0.1 GiB. Tune them on
   quality alone.
2. **GPS at graph width 64 is also free** -- 0.1 GiB more, for 2.1x the
   parameters (26.9M to 56.0M). The peak is dominated by the hidden-width
   correction, and the GPS stack's own tensors are too small at width 32 or 64
   to register beside it.
3. **GraNoLa's width is the one knob that costs.** Doubling graph width adds
   6.2 GiB, while every other GraNoLa setting is free. Its normalization keeps
   several graph-width tensors per token per block alive for the backward pass,
   so they scale with width where GPS's do not.
4. GPS depth adds exactly 1,168,384 parameters per block, linearly.

Each figure is one run. The 0.0 to 0.1 GiB differences are inside the noise of
a 15-second sampler, so read them as "no measurable cost", not as exact.

## The width formula holds for GPS

The microbatch reference claims peak memory is set by the product
`(token / subgraph) x graph microbatch`, and not by how that product is split.
Tested directly on GPS:

| Split | token | gm | width | peak GiB |
|---|---:|---:|---:|---:|
| A | 16000 | 28 | 224 | 61.4 |
| B | 8000 | 56 | 224 | 61.5 |

0.1 GiB apart. The claim extends to the GPS architecture, so a GPS run can be
tuned on either axis and only the product needs to be chosen.

### What this says so far

1. **GraNoLa is the cheapest normalization, by a wide margin.** At width 224 it
   peaks 12.0 GiB below batchnorm (52.4 against 64.4), and it is the only arm
   that completes at width 448. The reason is structural: batchnorm normalizes
   at hidden width 3584, GraNoLa normalizes at graph width 32 and only then
   projects out. More parameters, far less activation memory.
2. **GraNoLa `token` adaptivity costs more than `graph`.** Identical at width
   224, but `token` OOMs at 448 where `graph` completes, peaking 1.1 GiB higher
   before it dies. Per-token gamma and beta are the difference.
3. **Ceilings on a 95 GiB card**, training, this configuration:
   - batchnorm: between width 224 and 448
   - granola graph: between 448 and 896, completes at 448 at 92%
   - granola token: between 224 and 448
   - gps: between 224 and 448
4. **The cold-cache first job is the real constraint.** Batchnorm at width 128
   peaked at 77.9 GiB cold against 64.4 GiB at nearly double the width warm.
   A grid's first job pays for teacher scoring on top of everything else. Size
   the first job for the cold number, not the warm one.
5. **For a 48 GiB rtx_6000**, nothing here fits with headroom except gps at
   width 128 (43.5 GiB, and that is already 91% of 48). Expect to need width 64
   or below, or a smaller token microbatch. Untested — needs its own rungs.

## Wall time: one node died, and it explains the anomalies

21450082/83/84 all landed on `ise-6000p-01`, ran 11+ minutes for work that
takes 3 to 6 minutes, went silent in W&B, and were then requeued by Slurm with
their clocks reset. `sinfo` shows the cause:

```
ise-6000p-01 down* Not responding
```

The node was failing. I first read the slowness as ordinary contention between
three jobs sharing a node; that was wrong. A dying node explains all three
symptoms at once -- the stall, the lost W&B heartbeat, and the requeue.

**Consequences:**

1. Those three rungs have no valid measurement yet. Slurm requeued them, so
   they will produce one without intervention.
2. Their 38.3 GiB reading is from a machine that was already failing. Discard
   it rather than treat it as a low sample count.
3. Peak memory is per process, so a healthy neighbour does not distort it. Only
   the *sampling rate* suffers under load. Rungs with few samples on a healthy
   node are still worth re-running alone before their peak is trusted.
4. Wall times anywhere in this pilot remain weak evidence. They are single
   samples on a shared cluster, which is why the architecture speed comparison
   is not being drawn from them.
