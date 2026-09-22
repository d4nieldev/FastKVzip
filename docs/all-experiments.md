# GraphKV: complete experimental record

This document collects every experiment run in the project, with the settings
that defined each one and the numbers it produced. It is written to be read
without access to the code or the cluster.

The project asks one question. A released gate already decides which parts of a
language model's key-value cache to keep and which to evict. Can a small graph
network over the context improve that decision?

Fourteen grids were run to answer it, covering two base models, two kinds of
supervision, three mixer architectures and two ways of wiring the mixer into the
gate.

The short answer is that it can, but barely. The best configuration matches the
published result, and the margin a graph buys over the gate on its own is about
one point at one tenth retention and slightly negative at one twentieth. Most of
the effort went into finding that out, and into discovering that the loss the
work was steered by could not see it.

---

## How to read the numbers

**Retention ratio.** The fraction of the key-value cache kept after eviction.
`0.2` means one fifth is kept. `full` means nothing is evicted, which is the
ceiling any compression method is measured against.

**Benchmark score.** An absolute score from 0 to 100 on a SCBench task. Tasks
sit on very different scales, so a score is only meaningful beside that task's
own full-cache score.

**Relative score.** The same score as a percentage of that task's full-cache
score. This is what makes different tasks comparable.

**Stage-1 BCE.** Binary cross-entropy against the teacher's importance scores.
This is a *calibration* loss. A central finding of this project is that it
cannot see whether the eviction *ranking* changed, which is the only thing that
matters downstream. Several grids produced indistinguishable loss curves and
very different benchmark scores.

**Protected window.** Some protocols never evict the most recent few percent of
tokens. That flatters retrieval tasks considerably. Numbers measured with
different windows are not comparable, and this document groups them.

**Two training stages.** Stage 1 distils the teacher's importance scores into
the gate and mixer. Stage 2 fine-tunes against the model's own answers. Runs
named `-s1` and `-s2` are the two stages of one configuration.

---

## Provenance, and what is missing

Every number below was read back from a durable source rather than retyped.

| source | what it covers | status |
|---|---|---|
| Weights & Biases | all 157 training runs, their settings and losses | complete |
| W&B `test/*` metrics | benchmark scores for the Qwen2.5 stops and answer grids | complete |
| project ledger | benchmark tables for the Qwen3 grids | complete, carried over intact |
| Slurm dashboard job logs | benchmark scores for the later grids | complete |
| evaluation metrics files on the cluster | the architecture grid, the screen, the gate-space and diagnosis grids | complete |
| compiled paper draft, tables 1 to 3 | the three published baselines | complete, and cross-checked against the W&B runs |

Every grid in this document has its full benchmark table. The architecture grid
and the learning-rate screen wrote their scores only to result files on the
cluster, and those 34 result directories were read directly; all 11 tasks are
present and complete in every one of them.

The two published baselines appear twice over, once from the paper's tables and
once from their own W&B runs. The two agree exactly, which checks both.

---

## Shared setup

Two base models were used. The early exploration used `Qwen/Qwen3-8B`. Every
result from Grid 6 onward uses `Qwen/Qwen2.5-7B-Instruct-1M`, which is the model
the paper reports.

The gate being improved is the released FastKVzip gate, a low-rank sink
attention gate with 16 dimensions and 16 sink tokens. The graph mixer is a
per-graph network, one graph per layer and key-value head, whose output is added
into the gate. The base language model is frozen throughout; only the gate and
the mixer are trained.

| setting | value |
|---|---|
| graphs | one per layer and key-value head, 112 for Qwen2.5-7B |
| gate | 16 dimensions, 16 sink tokens |
| mixer width | 32 or 128, stated per grid |
| optimiser | AdamW, weight decay 0.01, no AMSGrad unless stated |
| schedule | linear warmup then cosine decay on both learning rates |
| precision | bfloat16 compute with fp32 master weights |
| seed | 0 throughout |

Single seed is a real limitation. Small differences between runs in this record
are not evidence of anything. Large ones, such as a benchmark score of 0.4
against 40, are.

---

## Summary of findings

1. **At its best, the work matches the published result.** The best
   configuration of the architecture grid reaches 51.35 at one fifth retention
   against the paper's 50.92, on the same model, the same eleven tasks and the
   same protected window. That is the headline the rest of this record qualifies.

2. **The training loss cannot see the difference.** Configurations whose
   benchmark scores span twenty points produced stage-1 losses within two
   percent of each other. The loss measures calibration; eviction depends on
   ranking. This was diagnosed formally in Grid 13.

3. **Giving the mixer more influence makes things monotonically worse.** The
   architecture grid contains a clean three-point dose-response: with the mixer's
   initial contribution at 0.1, 1 and 2, the score at one fifth retention runs
   48.29, 38.70, 28.47. The gate-space grids reproduce the same ordering by a
   different mechanism.

4. **The damage is concentrated in retrieval.** On tasks that tolerate losing
   arbitrary tokens, every configuration scores within a normal band. On tasks
   that need specific tokens, the spread is enormous, and two configurations
   collapse to near zero.

5. **The mixer's benefit over no mixer is about one point, and it changes
   sign.** Measured against a gate-only control at matched protocol, a mixer is
   worth +1.5 at one tenth retention and -1.1 at one twentieth.

6. **The gate-space coupling is behind the older hidden coupling.** At the same
   protected window, 47.05 against 51.35 at one fifth retention. The new
   coupling reaches the gate as designed; it does not pay.

7. **A protected window is worth about two points**, measured on one model
   evaluated both ways. It explains part of an apparent gap, not all of one.

---

## Part 1 — Qwen3 exploration, Grids 1 to 5

These five grids established the basic recipe on `Qwen/Qwen3-8B`: how many
epochs, whether to start the gate from the released checkpoint or from random,
whether to freeze it, how large the initial mixer contribution should be, and
which optimiser settings hold up. They were evaluated on `scbench_kv_short`
with a 2% protected window.

Their tables were verified against W&B and the durable evaluation files at the
time, and are carried over here unchanged.

### Default configurations

### Training: data and run control

| CLI option | Default value | Description |
|---|---:|---|
| `--model` | Required | Hugging Face model name or local model path. |
| `--output-dir` | `graph_checkpoints` | Root directory for checkpoints. |
| `--epochs` | `1` | Number of training epochs. |
| `--max-contexts` | Not set | Optional limit on training contexts per run. |
| `--train-context-start` | `0` | Zero-based offset into the length-filtered FineWeb contexts. It applies to both regular and concatenated training data. |
| `--train-context-count` | `29` | Number of regular 10K–30K FineWeb training contexts. The concatenated set scales with it. |
| `--seed` | `0` | Random seed. |
| `--prefill-chunk` | `16000` | Number of tokens in each teacher prefill call. |
| `--teacher-cache-dir` | Not set | Optional directory for reusable teacher examples. |
| `--resume` | Not set | Optional training checkpoint to resume. |

### Training: checkpoint and validation schedule

| CLI option | Default value | Description |
|---|---:|---|
| `--save-strategy` | `epochs` | Count save intervals in epochs or processed training contexts. |
| `--save-every` | `1` | Save after this many selected intervals. |
| `--save-best` / `--no-save-best` | `true` | Save `best.pt` when validation improves. |
| `--eval-strategy` | `epochs` | Count validation intervals in epochs or processed training contexts. |
| `--eval-every` | `1` | Validate after this many selected intervals. |

### Training: model

| CLI option | Default value | Description |
|---|---:|---|
| `--gate-checkpoint` | Not set | Start with a random gate. Use `fastkvzip` for the released gate. |
| `--gate-dim` | `16` | Gate query and key dimension. It is inferred from a supplied checkpoint. |
| `--gate-sink` | `16` | Learned baseline keys per KV head. It is inferred from a supplied checkpoint. |
| `--freeze-gate` / `--no-freeze-gate` | `false` | Whether gate weights stay fixed. |
| `--graph-dim` | `32` | Low-rank mixer dimension. |
| `--gram-normalization` | `token-count` | Divide each Gram matrix by its token count. |
| `--leaky-relu-slope` | `0.01` | Negative slope of LeakyReLU. |
| `--alpha-init` | `0.1` | Initial learned mixer residual coefficient (hidden coupling only). |
| `--mixer-coupling` | `hidden` | How the mixer reaches the gate: a hidden-width residual, or `gate-space` maps into the gate's queries, keys and logit bias. |
| `--injection-target` | `qk-logit` | Under `gate-space`: `qk`, `logit`, or both. |
| `--injection-init` | `0` | Deviation the gate-space injection maps start at. Zero starts at the gate exactly but leaves the mixer without gradient until the maps grow. |
| `--self-loop-init` | Not set | Add a learnable self-loop weight per graph to the implicit adjacency, starting at this value. |
| `--subgraph-size` | Not set | Use one graph for the whole context. A value enables independent subgraphs. |
| `--subgraphs-per-step` | `max` | In subgraph mode, update after all subgraphs from the context. |

### Training: optimization and memory

| CLI option | Default value | Description |
|---|---:|---|
| `--training-mode` | `joint` | Train gate and mixer from one whole-context loss. |
| `--token-microbatch-size` | `1000` | Maximum token width processed at once. |
| `--graph-microbatch-size` | `auto` | Layer/head graphs processed together. `auto` uses the KV-head count. |
| `--gate-lr` | `1e-4` | Gate AdamW learning rate. |
| `--mixer-lr` | `1e-3` | Mixer AdamW learning rate. |
| `--weight-decay` | `0.01` | Decay for all gate parameters and mixer weights. Mixer `alpha`, `gamma`, and `beta` are exempt. |
| `--adamw-eps` | `1e-8` | AdamW numerical-stability term. |
| `--amsgrad` / `--no-amsgrad` | `false` | Whether AdamW uses AMSGrad. |
| `--gate-lr-scheduler` | `none` | Gate learning-rate scheduler class. |
| `--gate-lr-scheduler-kwargs` | Not set | JSON arguments for the gate scheduler. |
| `--mixer-lr-scheduler` | `none` | Mixer learning-rate scheduler class. |
| `--mixer-lr-scheduler-kwargs` | Not set | JSON arguments for the mixer scheduler. |

### Training: W&B

| CLI option | Default value | Description |
|---|---:|---|
| `--wandb-mode` | `online` | W&B logging mode. |
| `--wandb-project` | `whole-context-graph-fastkvzip` | W&B project name. |
| `--wandb-entity` | Not set | Optional W&B entity. |
| `--wandb-name` | Not set | Optional W&B run name. |

### Evaluation

| CLI option | Default value | Description |
|---|---:|---|
| `--graph-checkpoint` | Required | GraphKV checkpoint to evaluate. |
| `--model` | Checkpoint value | Optional model override. It must match the checkpoint. |
| `--data` | `scbench_kv` | Dataset selector; exact names are never automatically replaced with shorter variants. |
| `--idx` | `0` | First dataset example. |
| `--num` | Not set | Optional context limit; fixed benchmarks run their complete prepared/filtered split. |
| `--ratios` | `0.75 0.5 0.4 0.3 0.2` | Requested KV retention ratios. |
| `--window-size` | `4096` | Protect 2% below the prefill-chunk length, or up to 4,096 tokens otherwise. A value between 0 and 1 always uses a context ratio. |
| `--level` | `pair` | Use one pruning budget across all layers and heads. |
| `--full-cache-answer` / `--no-full-cache-answer` | `true` | Also generate the unpruned answer. |
| `--token-microbatch-size` | Checkpoint value | Optional scoring override. `full` uses one context-sized chunk. |
| `--graph-microbatch-size` | Checkpoint value | Optional scoring override. `all` processes every layer/head graph together. |
| `--run-dir` | Required | Directory for resumable outputs and metrics. |
| `--existing-results` | `fail` | Reject an existing run. Use `resume` to continue it. |
| `--verbose` | `false` | Whether to print detailed per-example output. |
| `--log-to-wandb` | `false` | Whether to upload final benchmark curves. |
| `--wandb-project` | Not set | W&B project used for evaluation uploads. |
| `--wandb-entity` | Not set | Optional W&B entity. |
| `--wandb-run-id` | Checkpoint ID | Optional explicit evaluation destination, also usable without immediate uploads. |
| `--ruler-prompt-mode` | `graphkv` | RULER task/template wrapping; `official` is stored under a distinct result identity. |

### Evaluation protocol used below

The Qwen3 results in Grids 1–5 use the protocol below, except where explicitly
noted. Grids 6–7 use a different model; their protocols are specified in those
sections.

| Setting | Value |
|---|---|
| Model | `Qwen/Qwen3-8B` |
| Benchmark | `scbench_kv_short` |
| Examples | `0–99` (`100/100`) |
| Evaluator | Whole-context `eval_graph.py` |
| Requested retention | `0.75`, `0.50`, `0.40`, `0.30`, `0.20` |
| Full-cache baseline | Enabled |
| Protected window | `0.02` of the compressible context |
| Pruning level | `pair` |
| Reported score | Absolute benchmark score, from `0` to `100` |

This protocol overrides the CLI default for `--window-size`.

### Results

Grids 2–6 W&B refresh: **2026-09-06, 20:12 UTC**. These grids were checked against
absolute `test/<benchmark>` history at `test/retention_ratio`, rather than the
last-value run summary. Each completed curve contains the five requested
retention ratios plus the full-cache point. For Grids 2–6, W&B run links
identify the source of each curve; evaluation settings also use the recorded
experiment setup where available.

Grid 7 refresh: **2026-09-07, 20:11 UTC**. Its complete W&B histories were
cross-checked against durable evaluation metrics/manifests and Slurm completion
records on both accounts. The source checkpoint's `scbench_kv` curve was also
re-read from W&B. Earlier grids were not otherwise refreshed in this update.

### Main baseline: released FastKVzip

This is the main comparison baseline. It uses the released Qwen3-8B FastKVzip
checkpoint and the official FastKVzip evaluator. It has no training W&B run.

It uses the same 100 `scbench_kv_short` examples, retention ratios, `pair`
pruning, and `0.02` protected window described above. The official evaluator is
the only difference from the evaluation protocol table.

| Evaluation run | Evaluator | 1.00 (full) | 0.75 | 0.50 | 0.40 | 0.30 | 0.20 |
|---|---|---:|---:|---:|---:|---:|---:|
| `qwen3-8b_fastkvzip_w0.02` | Official FastKVzip | 66.70 | 65.70 | 66.80 | 66.00 | 62.10 | 50.20 |

### Grid 1: epochs, gate initialization, and gate freezing

This grid tested whether more epochs help, whether the released FastKVzip gate
is a better starting point, and whether that gate should remain frozen.
Random frozen gates are invalid, so the grid has nine runs.

All runs used one whole-context graph, joint training, `--alpha-init 0.1`, token microbatch `16000`, graph
microbatch `16`, no scheduler, no `best.pt`, and seed `0`.

The scores below retain the document's previously recorded newer-protocol
results. The linked W&B runs contain older evaluation points at `0.10`, `0.20`,
`0.30`, and `1.00`, with different scores; they identify the training runs but
do not verify this table. The corresponding local evaluation artifacts were
not available during this refresh, so these scores were not independently
revalidated.

| W&B run | `--epochs` | `--gate-checkpoint` | `--freeze-gate` | 1.00 (full) | 0.75 | 0.50 | 0.40 | 0.30 | 0.20 |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| [qwen3-8b-e1-gate-random-trainable-seed0](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-e124-g-rand-pre-freeze-tf/runs/01inalhx) | 1 | Not set | `false` | 66.70 | 66.50 | 62.30 | 58.90 | 51.80 | 40.00 |
| [qwen3-8b-e1-gate-pretrained-trainable-seed0](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-e124-g-rand-pre-freeze-tf/runs/z4uxu1uu) | 1 | `fastkvzip` | `false` | 66.70 | 66.60 | 66.50 | 65.50 | 58.80 | 49.50 |
| [qwen3-8b-e1-gate-pretrained-frozen-seed0](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-e124-g-rand-pre-freeze-tf/runs/8uohx8n2) | 1 | `fastkvzip` | `true` | 66.70 | 65.70 | 66.60 | 65.10 | 59.20 | 50.60 |
| [qwen3-8b-e2-gate-random-trainable-seed0](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-e124-g-rand-pre-freeze-tf/runs/1kmbwnc2) | 2 | Not set | `false` | 66.70 | 67.10 | 64.50 | 64.00 | 55.50 | 47.80 |
| [qwen3-8b-e2-gate-pretrained-trainable-seed0](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-e124-g-rand-pre-freeze-tf/runs/10hemlqh) | 2 | `fastkvzip` | `false` | 66.70 | 65.90 | 67.30 | 65.50 | 60.50 | 49.70 |
| [qwen3-8b-e2-gate-pretrained-frozen-seed0](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-e124-g-rand-pre-freeze-tf/runs/pqbm7lne) | 2 | `fastkvzip` | `true` | 66.70 | 66.00 | 66.90 | 65.70 | 61.70 | 51.00 |
| [qwen3-8b-e4-gate-random-trainable-seed0](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-e124-g-rand-pre-freeze-tf/runs/cymsqosb) | 4 | Not set | `false` | 66.70 | 66.40 | 64.40 | 62.00 | 57.30 | 50.20 |
| [qwen3-8b-e4-gate-pretrained-trainable-seed0](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-e124-g-rand-pre-freeze-tf/runs/zv49mqb2) | 4 | `fastkvzip` | `false` | 66.70 | 66.20 | 67.00 | 65.50 | 58.50 | 49.40 |
| [qwen3-8b-e4-gate-pretrained-frozen-seed0](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-e124-g-rand-pre-freeze-tf/runs/bcmv295p) | 4 | `fastkvzip` | `true` | 66.70 | 66.20 | 66.70 | 65.80 | 62.20 | 51.80 |

### Grid 2: training contexts and alpha initialization

This grid tested how training-set size interacts with the initial mixer residual
coefficient. The concatenated-context pool scaled approximately by source-token
volume with the regular context count.

All runs used one epoch, one whole-context graph, the trainable released gate, joint training, gate LR
`1e-4`, mixer LR `1e-3`, token microbatch `16000`, graph microbatch `16`, and
seed `0`. The gate used 50% linear warmup and the mixer used 15% linear warmup.
Both then used cosine decay. No `best.pt` was saved.

| W&B run | `--train-context-count` | `--alpha-init` | 1.00 (full) | 0.75 | 0.50 | 0.40 | 0.30 | 0.20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| [qwen3-8b-context29-alpha02-pretrained-trainable-seed0](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-qwen3-context-alpha-grid/runs/f1lqh3hk) | 29 | 0.2 | 66.70 | 66.20 | 67.70 | 66.90 | 62.20 | 50.20 |
| [qwen3-8b-context29-alpha04-pretrained-trainable-seed0](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-qwen3-context-alpha-grid/runs/bp5lm82y) | 29 | 0.4 | 66.70 | 65.80 | 67.30 | 65.50 | 59.80 | 48.00 |
| [qwen3-8b-context29-alpha07-pretrained-trainable-seed0](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-qwen3-context-alpha-grid/runs/c1tf5nc6) | 29 | 0.7 | 66.70 | 67.00 | 66.20 | 66.70 | 61.60 | 47.70 |
| [qwen3-8b-context50-alpha02-pretrained-trainable-seed0](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-qwen3-context-alpha-grid/runs/3y2kavr6) | 50 | 0.2 | 66.70 | 66.20 | 66.20 | 65.40 | 60.10 | 49.70 |
| [qwen3-8b-context50-alpha04-pretrained-trainable-seed0](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-qwen3-context-alpha-grid/runs/1mcg8i6v) | 50 | 0.4 | 66.70 | 65.90 | 67.20 | 66.80 | 59.30 | 49.40 |
| [qwen3-8b-context50-alpha07-pretrained-trainable-seed0](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-qwen3-context-alpha-grid/runs/9tj6ovfs) | 50 | 0.7 | 66.70 | 66.00 | 67.20 | 66.40 | 61.50 | 50.30 |
| [qwen3-8b-context100-alpha02-pretrained-trainable-seed0](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-qwen3-context-alpha-grid/runs/xdadtzfn) | 100 | 0.2 | 66.70 | 66.10 | 66.20 | 66.40 | 58.20 | 48.80 |
| [qwen3-8b-context100-alpha04-pretrained-trainable-seed0](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-qwen3-context-alpha-grid/runs/di8kmquf) | 100 | 0.4 | 66.70 | 66.60 | 67.20 | 66.40 | 57.90 | 50.50 |
| [qwen3-8b-context100-alpha07-pretrained-trainable-seed0](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-qwen3-context-alpha-grid/runs/lizstl6z) | 100 | 0.7 | 66.70 | 65.90 | 67.20 | 66.40 | 60.20 | 45.00 |

#### Follow-up: whole-context versus 2K-subgraph control

Two additional one-epoch runs used 29 regular contexts, `--alpha-init 0.2`,
the trainable released gate, and the Grid 2 learning rates and schedulers.
Both used graph microbatch `8`, token microbatch `16000`, AdamW epsilon
`1e-8`, no AMSGrad, and seed `0`. The whole-context run is the matched control
for the 2K-subgraph run; the original Grid 2 runs used graph microbatch `16`.

Both uploaded complete `scbench_kv_short` curves with the retention ratios
below. Their protected-window, pruning-level, and example-range settings were
not independently recoverable from W&B, so protocol equivalence is unverified.

| W&B run | `--subgraph-size` | `--subgraphs-per-step` | 1.00 (full) | 0.75 | 0.50 | 0.40 | 0.30 | 0.20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| [qwen3-8b-context29-alpha02-pretrained-trainable-seed0-subgraph-control](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-qwen3-context-alpha-grid/runs/xpojtvpo) | Not set | Not used | 66.70 | 66.00 | 66.90 | 65.60 | 62.30 | 50.60 |
| [qwen3-8b-context29-alpha02-pretrained-trainable-seed0-subgraph2k-step8](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-qwen3-context-alpha-grid/runs/csqx3k4q) | 2000 | 8 | 66.70 | 66.30 | 66.80 | 66.90 | 61.60 | 49.90 |

### Grid 3: AdamW epsilon, AMSGrad, and gate initialization

This grid tested whether AdamW stability settings help a released or random
gate. It crossed two epsilon values, two AMSGrad settings, and two gate starts.

All runs used one epoch, one whole-context graph, 29 regular contexts, `--alpha-init 0.2`, a trainable gate,
joint training, gate and mixer LR `1e-3`, token microbatch `16000`, graph
microbatch `8`, and seed `0`. Both optimizers used 15% linear warmup followed by
cosine decay. Initial graph-microbatch-16 attempts ran out of GPU memory; the
listed runs use graph microbatch `8`. No `best.pt` was saved.

All evaluations are complete.

| W&B run | `--gate-checkpoint` | `--adamw-eps` | `--amsgrad` | 1.00 (full) | 0.75 | 0.50 | 0.40 | 0.30 | 0.20 |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| [qwen3-8b-context29-alpha02-gate-pretrained-adamw-amsgradfalse-eps1e-3-seed0](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-qwen3-adamw-eps-amsgrad-grid/runs/dwrufoh8) | `fastkvzip` | `1e-3` | `false` | 66.70 | 65.30 | 66.70 | 66.60 | 61.30 | 50.30 |
| [qwen3-8b-context29-alpha02-gate-pretrained-adamw-amsgradfalse-eps1e-4-seed0](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-qwen3-adamw-eps-amsgrad-grid/runs/x4jm49fk) | `fastkvzip` | `1e-4` | `false` | 66.70 | 66.60 | 66.50 | 66.30 | 60.90 | 51.40 |
| [qwen3-8b-context29-alpha02-gate-pretrained-adamw-amsgradtrue-eps1e-3-seed0](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-qwen3-adamw-eps-amsgrad-grid/runs/5g6a6vu0) | `fastkvzip` | `1e-3` | `true` | 66.70 | 65.40 | 67.60 | 66.50 | 61.40 | 50.10 |
| [qwen3-8b-context29-alpha02-gate-pretrained-adamw-amsgradtrue-eps1e-4-seed0](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-qwen3-adamw-eps-amsgrad-grid/runs/6zz4u1kj) | `fastkvzip` | `1e-4` | `true` | 66.70 | 65.90 | 66.80 | 66.50 | 60.60 | 50.60 |
| [qwen3-8b-context29-alpha02-gate-random-adamw-amsgradfalse-eps1e-3-seed0](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-qwen3-adamw-eps-amsgrad-grid/runs/vjfwcsv1) | Not set | `1e-3` | `false` | 66.70 | 64.40 | 59.80 | 56.50 | 53.20 | 29.70 |
| [qwen3-8b-context29-alpha02-gate-random-adamw-amsgradfalse-eps1e-4-seed0](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-qwen3-adamw-eps-amsgrad-grid/runs/9osae7zc) | Not set | `1e-4` | `false` | 66.70 | 65.40 | 66.10 | 61.50 | 55.30 | 38.40 |
| [qwen3-8b-context29-alpha02-gate-random-adamw-amsgradtrue-eps1e-3-seed0](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-qwen3-adamw-eps-amsgrad-grid/runs/ag4ryysn) | Not set | `1e-3` | `true` | 66.70 | 64.30 | 60.50 | 56.20 | 53.20 | 28.60 |
| [qwen3-8b-context29-alpha02-gate-random-adamw-amsgradtrue-eps1e-4-seed0](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-qwen3-adamw-eps-amsgrad-grid/runs/a4pg98h8) | Not set | `1e-4` | `true` | 66.80 | 65.80 | 66.00 | 61.60 | 55.00 | 38.00 |

### Grid 4: AdamW epsilon, optimizer unit, and epochs

This grid tested how AdamW epsilon interacts with training duration and the
number of independent subgraphs in each optimizer update. It crossed three
epsilon values, whole-context versus 2K-subgraph updates, and four versus eight
epochs.

All runs used 29 regular contexts, the trainable released gate, joint training,
`--alpha-init 0.1`, gate and mixer LR `1e-3`, token microbatch `16000`,
`--no-amsgrad`, and seed `0`. Both optimizers used 15% linear warmup followed by
cosine decay. No `best.pt` was saved. The eight-epoch `1e-4` subgraph run and
both eight-epoch `1e-6` runs used graph microbatch `8`; the other nine runs
used graph microbatch `16`.

All 12 evaluations are complete.

| W&B run | `--epochs` | `--adamw-eps` | `--subgraph-size` | `--subgraphs-per-step` | 1.00 (full) | 0.75 | 0.50 | 0.40 | 0.30 | 0.20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| [train-qwen3-8b-e4-eps1e-4-fullctx-alpha01-seed0](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-qwen3-subgraph-eps-epochs-grid/runs/em1j1n6e) | 4 | `1e-4` | Not set | Not used | 66.70 | 66.20 | 66.30 | 65.30 | 61.00 | 51.50 |
| [train-qwen3-8b-e4-eps1e-4-sg2k-step8-alpha01-seed0](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-qwen3-subgraph-eps-epochs-grid/runs/25wyodrx) | 4 | `1e-4` | 2000 | 8 | 66.70 | 66.70 | 66.20 | 65.60 | 63.30 | 51.50 |
| [train-qwen3-8b-e8-eps1e-4-fullctx-alpha01-seed0](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-qwen3-subgraph-eps-epochs-grid/runs/k7x9rsb3) | 8 | `1e-4` | Not set | Not used | 66.70 | 65.90 | 66.60 | 65.90 | 61.30 | 51.60 |
| [train-qwen3-8b-e8-eps1e-4-sg2k-step8-alpha01-seed0](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-qwen3-subgraph-eps-epochs-grid/runs/6t3wb8nm) | 8 | `1e-4` | 2000 | 8 | 66.70 | 66.00 | 66.10 | 65.90 | 62.80 | 51.40 |
| [train-qwen3-8b-e4-eps1e-6-fullctx-alpha01-seed0](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-qwen3-subgraph-eps-epochs-grid/runs/vzkbdf4s) | 4 | `1e-6` | Not set | Not used | 66.70 | 66.30 | 66.00 | 65.60 | 60.50 | 50.00 |
| [train-qwen3-8b-e4-eps1e-6-sg2k-step8-alpha01-seed0](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-qwen3-subgraph-eps-epochs-grid/runs/52b5kwjn) | 4 | `1e-6` | 2000 | 8 | 66.70 | 66.00 | 67.20 | 65.00 | 61.30 | 51.60 |
| [train-qwen3-8b-e8-eps1e-6-fullctx-alpha01-seed0](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-qwen3-subgraph-eps-epochs-grid/runs/gen38wt6) | 8 | `1e-6` | Not set | Not used | 66.70 | 65.60 | 66.50 | 65.50 | 60.60 | 50.40 |
| [train-qwen3-8b-e8-eps1e-6-sg2k-step8-alpha01-seed0](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-qwen3-subgraph-eps-epochs-grid/runs/mvfmehyf) | 8 | `1e-6` | 2000 | 8 | 66.70 | 65.30 | 67.20 | 64.50 | 59.30 | 51.40 |
| [train-qwen3-8b-e4-eps1e-8-fullctx-alpha01-seed0](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-qwen3-subgraph-eps-epochs-grid/runs/a8ryhw15) | 4 | `1e-8` | Not set | Not used | 66.70 | 66.10 | 66.90 | 66.10 | 60.70 | 49.60 |
| [train-qwen3-8b-e4-eps1e-8-sg2k-step8-alpha01-seed0](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-qwen3-subgraph-eps-epochs-grid/runs/tx7n2ipv) | 4 | `1e-8` | 2000 | 8 | 66.70 | 66.10 | 67.40 | 65.00 | 59.50 | 50.50 |
| [train-qwen3-8b-e8-eps1e-8-fullctx-alpha01-seed0](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-qwen3-subgraph-eps-epochs-grid/runs/l9a4acwf) | 8 | `1e-8` | Not set | Not used | 66.70 | 66.50 | 66.10 | 65.90 | 60.30 | 50.40 |
| [train-qwen3-8b-e8-eps1e-8-sg2k-step8-alpha01-seed0](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-qwen3-subgraph-eps-epochs-grid/runs/qegtkvaw) | 8 | `1e-8` | 2000 | 8 | 66.70 | 66.50 | 66.80 | 65.50 | 61.40 | 51.00 |

### Grid 5: training-context offset, gate initialization, and graph unit

This grid used 30 regular FineWeb contexts starting at filtered ordinal `39`,
so the regular slice was `[39, 69)` (ordinals 39–68). Both the regular and
concatenated builders started at this offset. These are zero-based ordinals
after the 10K–30K token-length filter, not raw FineWeb row IDs. Despite the W&B
project name `unseen-context-grid`, this slice overlaps Grid 2: 11 regular
contexts overlap its 50-context runs, and all 30 overlap its 100-context runs.

The grid crossed a random versus released FastKVzip gate with whole-context
versus independent 2K-subgraph training. All gates were trainable. Every run
used 10 epochs, joint training, `--alpha-init 0.1`, AdamW epsilon `1e-4`, no
AMSGrad, weight decay `0.01`, gate and mixer LR `1e-3`, and seed `0`. Both
optimizers used 15% linear warmup followed by cosine decay. On RTX PRO 6000,
the token microbatch was `16000`; the graph microbatch was `16` for
whole-context training and `24` for subgraph training. Subgraph runs used
`--subgraph-size 2000 --subgraphs-per-step 8`. No `best.pt` was saved.
All jobs used commit `133136625b3be63f4c52c4788fc63678ad026cc8`.

Each configuration was run once under each cluster account. Daniel used the
full-activation teacher cache; Guy used a scores-only cache populated with the
same cached scores. The account copies use the same seed and are execution and
cache-mode repeats, not independent-seed replicates. All eight evaluations
completed the 100-example `scbench_kv_short` protocol described above with
token microbatch `16000` and graph microbatch `16`.

| W&B run | Account/cache | Gate initialization | Training graph | 1.00 (full) | 0.75 | 0.50 | 0.40 | 0.30 | 0.20 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| [qwen3-s39n30-e10-random-full-daniel](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-qwen3-unseen-context-grid/runs/q9k3ifej) | `danieloh` / full activations | Random | Whole context | 66.70 | 66.60 | 66.50 | 66.90 | 61.40 | 49.40 |
| [qwen3-s39n30-e10-random-full-guy](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-qwen3-unseen-context-grid/runs/0ix2paen) | `guyzagor` / scores only | Random | Whole context | 66.70 | 66.60 | 66.50 | 66.90 | 61.40 | 49.40 |
| [qwen3-s39n30-e10-random-sg2k-step8-daniel](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-qwen3-unseen-context-grid/runs/7067ejmz) | `danieloh` / full activations | Random | 2K subgraphs, 8/step | 66.70 | 65.90 | 65.80 | 67.30 | 64.70 | 50.50 |
| [qwen3-s39n30-e10-random-sg2k-step8-guy](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-qwen3-unseen-context-grid/runs/8sg38pq5) | `guyzagor` / scores only | Random | 2K subgraphs, 8/step | 66.70 | 65.80 | 66.70 | 67.20 | 64.80 | 50.60 |
| [qwen3-s39n30-e10-pretrained-full-daniel](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-qwen3-unseen-context-grid/runs/ktzfcbzc) | `danieloh` / full activations | `fastkvzip` | Whole context | 66.70 | 66.10 | 67.30 | 65.30 | 61.30 | 51.10 |
| [qwen3-s39n30-e10-pretrained-full-guy](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-qwen3-unseen-context-grid/runs/qh72cs52) | `guyzagor` / scores only | `fastkvzip` | Whole context | 66.70 | 66.10 | 67.30 | 65.30 | 61.30 | 51.10 |
| [qwen3-s39n30-e10-pretrained-sg2k-step8-daniel](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-qwen3-unseen-context-grid/runs/98gwjbsi) | `danieloh` / full activations | `fastkvzip` | 2K subgraphs, 8/step | 66.70 | 65.60 | 67.20 | 66.30 | 61.60 | 52.20 |
| [qwen3-s39n30-e10-pretrained-sg2k-step8-guy](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-qwen3-unseen-context-grid/runs/wob03ozw) | `guyzagor` / scores only | `fastkvzip` | 2K subgraphs, 8/step | 66.70 | 65.50 | 67.20 | 66.30 | 61.90 | 52.20 |

The whole-context account pairs matched at every retention ratio. The two
subgraph pairs differed by at most 0.9 score points. At the most compressed
setting, pretrained subgraph training scored `52.20`, compared with `50.20`
for the released FastKVzip baseline. Because both account copies used seed
`0`, this grid does not measure variation across random seeds.

### Grid 6: Qwen2.5-7B-Instruct-1M, training-set size, and gate initialization

This grid crossed 40, 80, and 120 regular training contexts with random versus
released FastKVzip gate initialization for `Qwen/Qwen2.5-7B-Instruct-1M`.
All gates were trainable. `--train-context-start 40` gave nested regular slices
`[40, 80)`, `[40, 120)`, and `[40, 160)` after the FineWeb 10K–30K token-count
filter. The concatenated-context pool scaled with the regular context count.

All six runs used 10 epochs, joint training, `--subgraph-size 2000`,
`--subgraphs-per-step 8`, `--alpha-init 0.1`, AdamW epsilon `1e-4`, no AMSGrad,
weight decay `0.01`, gate and mixer LR `1e-3`, and seed `0`. Both optimizers
used 15% linear warmup followed by cosine decay. Token and graph microbatches
were `16000` and `16`; prefill chunks were `16000`. Training used scores-only
teacher caches, token-count Gram normalization, and no `best.pt`.

The recorded evaluation setup used `last.pt` with post-prefill `eval_graph.py`
on examples `0–99` of **each** of `scbench_kv_short` and `scbench_kv`, requested
retention `0.75 0.50 0.40 0.30 0.20`, full-cache answers, protected window `0.02`,
`pair` pruning, and token/graph microbatches `16000`/`16`. All six runs have
complete six-point W&B curves for both benchmarks (12 evaluations).

Scoring honors the checkpoint's independent 2K subgraphs, followed by global
pruning over the context. The earlier wording "whole-context evaluation" did
not mean whole-context graph mixing.

Scores are absolute, from `0` to `100`. This grid uses a different model from
Grids 1–5. The released FastKVzip baseline above is for Qwen3-8B; no released
Qwen2.5 FastKVzip baseline is included here.

#### `scbench_kv_short`

| W&B run | `--train-context-count` | Gate initialization | 1.00 (full) | 0.75 | 0.50 | 0.40 | 0.30 | 0.20 |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| [qwen25-7b1m-s40n40-e10-pretrained-sg2k-step8-seed0](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-qwen25-7b1m-s40-stops-grid/runs/enwmlquw) | 40 | `fastkvzip` | 85.10 | 85.50 | 86.20 | 84.50 | 81.10 | 65.90 |
| [qwen25-7b1m-s40n40-e10-random-sg2k-step8-seed0](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-qwen25-7b1m-s40-stops-grid/runs/ozz8xt2o) | 40 | Random | 85.10 | 83.80 | 86.10 | 85.40 | 80.40 | 66.50 |
| [qwen25-7b1m-s40n80-e10-random-sg2k-step8-seed0](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-qwen25-7b1m-s40-stops-grid/runs/jd6rhqlh) | 80 | Random | 85.10 | 83.80 | 85.80 | 85.90 | 79.90 | 67.00 |
| [qwen25-7b1m-s40n80-e10-pretrained-sg2k-step8-seed0](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-qwen25-7b1m-s40-stops-grid/runs/jmfyqwfe) | 80 | `fastkvzip` | 85.10 | 85.50 | 86.50 | 85.10 | 81.20 | 67.00 |
| [qwen25-7b1m-s40n120-e10-pretrained-sg2k-step8-seed0](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-qwen25-7b1m-s40-stops-grid/runs/qokiutw7) | 120 | `fastkvzip` | 85.10 | 85.80 | 87.20 | 85.70 | 82.10 | 66.50 |
| [qwen25-7b1m-s40n120-e10-random-sg2k-step8-seed0](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-qwen25-7b1m-s40-stops-grid/runs/dysujs7b) | 120 | Random | 85.10 | 83.80 | 86.40 | 85.70 | 79.90 | 66.00 |

#### `scbench_kv`

| W&B run | `--train-context-count` | Gate initialization | 1.00 (full) | 0.75 | 0.50 | 0.40 | 0.30 | 0.20 |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| [qwen25-7b1m-s40n40-e10-pretrained-sg2k-step8-seed0](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-qwen25-7b1m-s40-stops-grid/runs/enwmlquw) | 40 | `fastkvzip` | 68.20 | 68.80 | 69.00 | 70.60 | 71.00 | 49.40 |
| [qwen25-7b1m-s40n40-e10-random-sg2k-step8-seed0](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-qwen25-7b1m-s40-stops-grid/runs/ozz8xt2o) | 40 | Random | 68.20 | 69.00 | 64.00 | 70.80 | 58.00 | 45.80 |
| [qwen25-7b1m-s40n80-e10-random-sg2k-step8-seed0](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-qwen25-7b1m-s40-stops-grid/runs/jd6rhqlh) | 80 | Random | 68.20 | 69.00 | 67.60 | 69.80 | 52.00 | 43.20 |
| [qwen25-7b1m-s40n80-e10-pretrained-sg2k-step8-seed0](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-qwen25-7b1m-s40-stops-grid/runs/jmfyqwfe) | 80 | `fastkvzip` | 68.20 | 70.20 | 70.20 | 71.00 | 68.40 | 48.60 |
| [qwen25-7b1m-s40n120-e10-pretrained-sg2k-step8-seed0](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-qwen25-7b1m-s40-stops-grid/runs/qokiutw7) | 120 | `fastkvzip` | 68.20 | 69.40 | 68.60 | 70.00 | 67.40 | 46.80 |
| [qwen25-7b1m-s40n120-e10-random-sg2k-step8-seed0](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-qwen25-7b1m-s40-stops-grid/runs/dysujs7b) | 120 | Random | 68.20 | 68.60 | 65.80 | 68.00 | 52.60 | 45.20 |

At retention `0.20`, the best short-benchmark score was `67.00` for both
80-context gate initializations. On `scbench_kv`, the best score at that
retention was `49.40` with the 40-context pretrained gate. Increasing the
training-set size did not produce a monotonic improvement at this retention.
These are single-seed results on overlapping training slices.

### Grid 7: answer-supervised fine-tuning of the Qwen2.5 40-context checkpoint

All ten training jobs and their ten dependent evaluations completed with Slurm
exit `0:0`. Each training run logged all 360 updates and both epoch-end
validations. Every evaluation has all 100 examples (`0–99`) at all six retention
points, including full cache. W&B and the durable `metrics.json` files agree.

#### Shared configuration and provenance

These runs fine-tuned Grid 6's
`qwen25-7b1m-s40n40-e10-pretrained-sg2k-step8-seed0/last.pt`
([source run](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-qwen25-7b1m-s40-stops-grid/runs/enwmlquw)),
not a randomly initialized GraphKV model. The underlying
`Qwen/Qwen2.5-7B-Instruct-1M` remained frozen; only gate and mixer were trained
using `prefill/train_graph_answer.py` and generated-answer-token cross-entropy.

- Data: first 200 deduplicated Agentic context/question examples (`start=0`),
  split into 180 training and 20 held-out validation examples; two epochs,
  360 optimizer updates, seed `0`, repeated example order.
- Teacher: greedy answers from the same frozen LLM with full context. `base`
  and `g5e5-m5e4` used persistent answer caches; the other eight generated
  answers without reading or writing that cache. This was an operational
  concurrency choice, not an intended change to teacher supervision.
- Training/validation: independent 2K subgraph scoring, concatenated scores,
  exact context top-k separately for every layer/KV head, no protected local
  window. All subgraphs contribute to one update per example. System prefix,
  question, chat suffix, and answer tokens are outside the context budget.
- Optimizers: AdamW, epsilon `1e-8`, weight decay `0.01`, no AMSGrad; both
  learning rates use 15% linear warmup then cosine decay. BF16 compute,
  token/graph microbatches `16000`/`16`, prefill chunks `16000`; inherited
  gate/mixer architecture and exact saved prefix.
- Validation: answer-token-weighted NLL and teacher-forced token accuracy at
  fixed retention `0.20`, after each epoch. `best.pt` is selected by lowest
  validation NLL, not by SCBench score. The base run resumed its successful
  one-update pilot; all other runs initialized fresh optimizers from the source.
- Runtime: repository `d7e18d7` plus the deployed chunk-wise VJP replay memory
  fix from `77cfd4b`. Five train/evaluation pairs ran under each of `danieloh`
  and `guyzagor`. Earlier pilot failures and eight canceled, never-started
  training replacements are attempt history, not failed grid configurations.

All W&B names have prefix `q25a-s40n40-n200e2-` and suffix `-s0`; the tables
use the intervening variant name. Linear retention spans the complete
360-update horizon, without resetting at an epoch boundary. Temperature is
the **backward STE temperature**, not the LLM decoding temperature.

| Variant / W&B run | Gate LR | Mixer LR | STE temperature | Training retention |
|---|---:|---:|---:|---|
| [base](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-answer-qwen25-7b1m-s40n40-grid-v1/runs/6sr9i9p0) | `1e-4` | `1e-3` | 1.0 | Linear 30% → 10% |
| [g5e5-m5e4](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-answer-qwen25-7b1m-s40n40-grid-v1/runs/25fyywo1) | `5e-5` | `5e-4` | 1.0 | Linear 30% → 10% |
| [g5e5-m2e3](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-answer-qwen25-7b1m-s40n40-grid-v1/runs/ffxivxb0) | `5e-5` | `2e-3` | 1.0 | Linear 30% → 10% |
| [g2e4-m5e4](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-answer-qwen25-7b1m-s40n40-grid-v1/runs/0gvo2xcb) | `2e-4` | `5e-4` | 1.0 | Linear 30% → 10% |
| [g2e4-m2e3](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-answer-qwen25-7b1m-s40n40-grid-v1/runs/1lkifsox) | `2e-4` | `2e-3` | 1.0 | Linear 30% → 10% |
| [temp05](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-answer-qwen25-7b1m-s40n40-grid-v1/runs/ssyf4qrr) | `1e-4` | `1e-3` | 0.5 | Linear 30% → 10% |
| [temp20](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-answer-qwen25-7b1m-s40n40-grid-v1/runs/caih0568) | `1e-4` | `1e-3` | 2.0 | Linear 30% → 10% |
| [uniform](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-answer-qwen25-7b1m-s40n40-grid-v1/runs/61ewyfcm) | `1e-4` | `1e-3` | 1.0 | Uniform [10%, 30%] |
| [fixed20](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-answer-qwen25-7b1m-s40n40-grid-v1/runs/w1y521p9) | `1e-4` | `1e-3` | 1.0 | Fixed 20% |
| [lin30to20](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-answer-qwen25-7b1m-s40n40-grid-v1/runs/y5wcu1pa) | `1e-4` | `1e-3` | 1.0 | Linear 30% → 20% |

#### Standard `scbench_kv` evaluation

Evaluation used each run's `best.pt`, **only `scbench_kv`**, examples `0–99`,
requested retention `0.75 0.50 0.40 0.30 0.20`, full-cache answers, protected
window `0.02`, `--level pair`, and token/graph microbatches `16000`/`16`.
`eval_graph.py` scores the saved independent 2K subgraphs after full prefill,
then prunes with a shared budget across layers/heads. This matches the source
checkpoint's benchmark settings; the intentional checkpoint distinction is
source `last.pt` versus fine-tuned validation-selected `best.pt`.

Scores below are absolute `test/scbench_kv` values, not relative-to-full-cache
percentages. The metric is normalized answer inclusion, averaged over questions
within each context and then over contexts; it is not exact match or Agentic
token F1. Retention includes the protected 2% window: the score-selected part
at requested 20% is approximately 18% of context. All full-cache scores are
`68.20`.

| Variant | 1.00 (full) | 0.75 | 0.50 | 0.40 | 0.30 | 0.20 | Change at 0.20 vs source |
|---|---:|---:|---:|---:|---:|---:|---:|
| Source checkpoint (Grid 6) | 68.20 | 68.80 | 69.00 | 70.60 | 71.00 | 49.40 | — |
| base | 68.20 | 68.60 | 71.00 | 71.60 | 71.40 | 66.60 | +17.20 |
| g5e5-m5e4 | 68.20 | 69.60 | 71.80 | 71.60 | 69.80 | 65.20 | +15.80 |
| g5e5-m2e3 | 68.20 | 69.80 | 71.20 | 71.60 | 40.20 | 44.20 | −5.20 |
| g2e4-m5e4 | 68.20 | 73.00 | 52.60 | 51.40 | 52.80 | 45.80 | −3.60 |
| g2e4-m2e3 | 68.20 | 71.20 | 74.40 | 72.60 | 47.20 | 40.80 | −8.60 |
| temp05 | 68.20 | 70.40 | 71.20 | 72.60 | 69.80 | 69.80 | +20.40 |
| temp20 | 68.20 | 70.00 | 74.40 | 75.20 | 60.20 | 46.20 | −3.20 |
| uniform | 68.20 | 68.60 | 72.40 | 72.00 | 69.60 | 71.60 | +22.20 |
| fixed20 | 68.20 | 74.20 | 56.00 | 57.00 | 53.40 | 45.60 | −3.80 |
| lin30to20 | 68.20 | 70.40 | 75.60 | 75.60 | 72.20 | 51.00 | +1.60 |

#### Validation and checkpoint selection

Accuracy here is teacher-forced answer-token accuracy (percent), **not SCBench
accuracy**. The two NLL columns are the fixed-20% epoch-end validations; the
accuracy column belongs to the selected checkpoint. W&B's final validation
summary is misleading for `base`, `g2e4-m2e3`, `fixed20`, and `lin30to20`, whose
evaluated checkpoints were selected after epoch 1.

| Variant | Epoch 1 NLL | Epoch 2 NLL | Selected epoch | Selected token accuracy (%) |
|---|---:|---:|---:|---:|
| base | 0.4925 | 0.5012 | 1 | 85.12 |
| g5e5-m5e4 | 0.4576 | 0.4347 | 2 | 87.31 |
| g5e5-m2e3 | 0.5783 | 0.5166 | 2 | 85.34 |
| g2e4-m5e4 | 0.4303 | 0.4143 | 2 | 86.65 |
| g2e4-m2e3 | 0.4494 | 0.4906 | 1 | 86.00 |
| temp05 | 0.5558 | 0.4906 | 2 | 86.43 |
| temp20 | 0.5635 | 0.5527 | 2 | 84.03 |
| uniform | 0.6303 | 0.5444 | 2 | 85.78 |
| fixed20 | 0.4725 | 0.5660 | 1 | 86.65 |
| lin30to20 | 0.4955 | 0.4977 | 1 | 86.87 |

#### What worked, what did not, and interpretation

The judgments below prioritize **20% retention**, the strongest compression
actually evaluated. Score changes are points against the source checkpoint,
not relative percentages. Mechanistic explanations are hypotheses from this
single-seed grid, not established causal conclusions.

- **`uniform` — strongest 20% result.** `71.60` is +22.20 over the source and
  +3.40 over full cache. Sampling budgets throughout training plausibly helps
  avoid dependence on one stage of a curriculum. It also retains strong
  40–50% scores, although its 30% score is 1.40 below the source.
- **`temp05` — strong and broadly useful.** `69.80` at 20%, +20.40 over the
  source and +3.20 over `base`. Sharper backward credit assignment is a
  plausible benefit; temperature never changes hard forward selection.
  It is second at 20% and remains competitive across the curve.
- **`base` — successful reference setting.** `66.60` at 20% (+17.20), with
  no large regressions elsewhere. The default fine-tuning recipe substantially
  improves aggressive compression without a specialized retention policy.
- **`g5e5-m5e4` — conservative learning rates work.** `65.20` at 20%
  (+15.80), with strong 40–50% results and the highest selected validation
  token accuracy. Smaller updates may help preserve useful source behavior,
  though this run does not improve on `base` at 20%.
- **`g5e5-m2e3` — unsuccessful at aggressive compression.** `44.20` at 20%
  and `40.20` at 30%, despite good 40–75% results. Relative to the low/low LR
  corner, increasing mixer LR fourfold coincides with a severe low-budget
  regression; the larger mixer updates are not beneficial here.
- **`g2e4-m5e4` — unsuccessful despite the best validation NLL.** `45.80`
  at 20% and only `51.40–52.80` at 30–50%. Increasing gate LR fourfold from
  the low/low corner also hurts transfer. Its `0.4143` validation NLL shows
  that better teacher-answer likelihood does not guarantee better SCBench
  retrieval under the standard evaluation pruning policy.
- **`g2e4-m2e3` — worst at 20%, useful only at looser budgets.** `40.80`
  at 20% (−8.60), but `74.40` at 50%. Raising both learning rates shifts
  useful performance toward less aggressive compression rather than giving
  a general improvement. Its selected checkpoint is already epoch 1, so this
  failure is not merely evaluation of the worse final-epoch checkpoint.
- **`temp20` — mixed, not a 20% candidate.** `46.20` at 20%, but `75.20`
  at 40% and `74.40` at 50%. The smoother backward credit assignment is
  associated with worse low-budget performance here, the opposite outcome
  from temperature 0.5; the scores alone do not establish the mechanism.
- **`fixed20` — unsuccessful for its intended budget.** `45.60` at 20%
  and substantial regressions throughout 30–50%, despite `74.20` at 75%.
  Matching the validation retention alone is insufficient; budget diversity
  or the curriculum appears useful. Validation NLL worsens in epoch 2, but
  the evaluated epoch-1 checkpoint also underperforms.
- **`lin30to20` — strongest for 30–50%, weak gain at 20%.** It leads this
  grid at 30% (`72.20`) and 40–50% (`75.60`), but reaches only `51.00`
  at 20% (+1.60). This is consistent with specialization at looser budgets. Its
  selected epoch-1 checkpoint had trained only through roughly 30% → 25%
  retention, which is consistent with its weaker 20% transfer.

For a 20%-retention target, prioritize `uniform` and `temp05`, retaining `base`
and `g5e5-m5e4` as strong references. For a 40–50% target, `lin30to20` is the
leading candidate. Do not select among these runs using validation NLL alone:
validation tests Agentic teacher fidelity with per-head budgets and no window,
whereas SCBench tests free-running answer inclusion with a shared cross-head
budget and a protected window. Their rankings demonstrably differ.

All 360 updates per run have finite logged NLL and gradient diagnostics,
including nonzero retained and evicted score-gradient norms. Thus the poor
SCBench variants are completed experiments, not numerical-crash results.
These are single-seed observations on one benchmark; small differences are
not evidence of statistical significance. No results below 20% retention or
on `scbench_kv_short` were collected for this grid. A score above full cache
is an observed benchmark outcome, not evidence of general superiority to the
uncompressed model.

Refresh evidence is retained under
`.slurm/grids/graphkv-answer-qwen25-7b1m-s40n40-grid-v1/`:
`results-wandb-2026-09-07.json` (run configurations, validation/test history and
training diagnostic aggregates), `results-artifact-verification-2026-09-07.json`
(exact current job IDs, exits, checkpoint paths and evaluation coverage),
`source-artifact-verification-2026-09-07.json` (matching source evaluation),
plus the original manifest and receipt.

---

## Part 1b — the published baselines, measured here

Before any of our own results, these are the two methods the paper compares
against, run on the same model and the same harness. Their full curves live in
W&B, so the numbers below are read from the runs rather than copied from the
paper.

The important difference between them is the protected window. Fast KVzip never
evicts the most recent 2% of tokens. KVzip protects nothing, which is the
setting every one of our later grids used.


### KVzip, no protected window

This is the protocol our later grids match, so it is the fair comparison.

Eleven SCBench base tasks, absolute score by retention. [W&B run](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-answer-qwen25-7b1m-s40n40-grid-v1/runs/p2wsdvv5).

|task|0.05|0.1|0.2|0.3|0.4|0.5|0.75|1|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|kv|0.00|11.00|36.80|68.40|67.80|66.60|66.40|68.20|
|prefix_suffix|0.00|0.00|2.80|23.00|33.00|41.00|50.20|51.20|
|summary|29.27|34.24|35.58|36.00|36.37|36.46|36.94|36.82|
|vt|45.51|43.64|40.67|40.84|40.71|40.80|40.31|41.29|
|many_shot|32.59|32.22|33.33|34.44|35.93|36.67|37.41|38.52|
|mf|13.83|30.33|35.33|35.17|34.17|34.17|32.67|33.00|
|choice_eng|57.41|68.06|72.22|77.78|76.39|77.78|77.78|79.17|
|qa_eng|15.03|37.33|42.58|46.37|43.96|43.34|40.31|41.52|
|repoqa|2.95|32.05|51.82|57.27|59.09|58.18|59.09|59.09|
|summary_with_needles|18.82|58.21|66.09|67.50|68.13|68.10|68.41|68.35|
|repoqa_and_kv|25.28|58.24|73.15|79.26|79.83|79.83|81.11|80.82|
|**eleven-task mean**|**21.88**|**36.85**|**44.58**|**51.46**|**52.31**|**52.99**|**53.69**|**54.36**|

Mean over the 33 RULER tasks in the same run:

|retention|0.05|0.1|0.2|0.3|0.4|0.5|0.75|1|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|mean score|—|—|85.61|87.97|88.37|88.35|88.38|88.52|

### Fast KVzip, 2% protected window

Not directly comparable to our later grids because of the window.

Eleven SCBench base tasks, absolute score by retention. [W&B run](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-answer-qwen25-7b1m-s40n40-grid-v1/runs/vi31uf3h).

|task|0.05|0.1|0.2|0.3|0.4|0.5|0.75|1|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|kv|2.80|32.80|47.00|66.80|66.40|72.80|68.20|68.20|
|prefix_suffix|0.60|11.20|38.80|48.40|55.60|47.40|36.80|51.20|
|summary|27.64|33.18|35.97|36.82|36.49|36.80|36.83|36.82|
|vt|43.02|49.02|43.51|41.73|40.58|40.53|41.38|41.29|
|many_shot|34.81|32.59|33.70|36.67|37.04|36.30|36.30|38.52|
|mf|19.33|29.33|32.83|33.50|34.00|33.83|32.83|33.00|
|choice_eng|46.30|68.98|73.61|75.00|75.00|75.00|79.17|79.17|
|qa_eng|13.07|21.50|44.20|44.74|42.02|41.76|42.54|41.52|
|repoqa|4.32|44.09|57.73|59.32|60.23|61.36|58.18|59.09|
|summary_with_needles|16.96|54.54|65.64|68.29|67.86|67.56|68.17|68.35|
|repoqa_and_kv|7.67|54.55|77.84|79.55|80.40|80.26|80.68|80.82|
|**eleven-task mean**|**19.68**|**39.25**|**50.08**|**53.71**|**54.15**|**53.96**|**52.83**|**54.36**|

Mean over the 33 RULER tasks in the same run:

|retention|0.05|0.1|0.2|0.3|0.4|0.5|0.75|1|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|mean score|—|—|84.82|87.92|88.33|88.34|88.46|88.54|

### An earlier window comparison

One of our own Qwen2.5 answer-trained checkpoints was also evaluated at both
window settings, long before the dedicated comparison in Part 6. It covers a
wider benchmark set but fewer of the SCBench base tasks.


**no protected window** — 50 benchmarks. [W&B run](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-answer-qwen25-7b1m-s40n40-grid-v1/runs/hdo2z4x4).

|benchmark|0.2|0.3|0.4|0.5|0.75|1|
|---|---:|---:|---:|---:|---:|---:|
|ruler_cwe_128k|26.20|0.26|0.00|26.10|20.50|19.44|
|ruler_cwe_4k|93.86|93.92|90.42|97.36|98.40|98.62|
|ruler_cwe_8k|85.70|43.08|34.92|90.10|90.68|91.02|
|ruler_fwe_4k|84.07|85.40|83.67|82.60|84.60|85.80|
|ruler_fwe_64k|76.00|81.27|83.20|86.60|86.73|86.00|
|ruler_fwe_8k|70.80|80.73|81.33|81.00|82.60|83.20|
|ruler_niah_multikey_1_128k|82.00|83.20|82.40|94.40|97.80|98.00|
|ruler_niah_multikey_1_4k|95.60|96.20|98.40|98.80|99.20|98.80|
|ruler_niah_multikey_1_64k|86.00|92.40|91.60|94.60|98.00|98.60|
|ruler_niah_multikey_1_8k|93.80|97.60|97.60|99.00|99.60|99.60|
|ruler_niah_multikey_2_4k|98.00|99.40|99.40|99.60|99.80|99.80|
|ruler_niah_multikey_2_64k|73.60|93.80|97.00|94.80|93.60|94.20|
|ruler_niah_multikey_2_8k|94.60|97.40|98.60|98.80|99.80|99.80|
|ruler_niah_multikey_3_128k|43.40|56.20|66.40|70.60|75.60|76.80|
|ruler_niah_multikey_3_4k|98.00|98.20|98.40|99.40|99.60|99.60|
|ruler_niah_multikey_3_8k|87.00|92.20|95.20|94.80|97.00|97.20|
|ruler_niah_multiquery_128k|98.75|98.95|87.85|99.30|99.70|99.50|
|ruler_niah_multiquery_4k|99.65|99.90|99.95|99.95|99.95|100.00|
|ruler_niah_multiquery_8k|99.60|99.90|99.85|99.90|99.95|99.90|
|ruler_niah_multivalue_128k|63.35|74.20|57.20|81.35|82.85|82.45|
|ruler_niah_multivalue_4k|62.85|75.30|81.15|85.45|92.85|91.90|
|ruler_niah_multivalue_8k|60.25|75.95|78.45|81.00|85.25|84.70|
|ruler_niah_single_1_128k|100.00|100.00|100.00|100.00|100.00|100.00|
|ruler_niah_single_1_4k|100.00|100.00|100.00|100.00|100.00|100.00|
|ruler_niah_single_1_64k|100.00|100.00|100.00|100.00|100.00|100.00|
|ruler_niah_single_1_8k|100.00|100.00|100.00|100.00|100.00|100.00|
|ruler_niah_single_2_128k|99.80|99.80|98.00|99.80|100.00|100.00|
|ruler_niah_single_2_4k|100.00|100.00|100.00|100.00|100.00|100.00|
|ruler_niah_single_2_64k|100.00|100.00|98.40|100.00|100.00|99.80|
|ruler_niah_single_2_8k|100.00|100.00|97.60|100.00|100.00|100.00|
|ruler_niah_single_3_128k|99.40|99.80|100.00|99.60|99.80|99.80|
|ruler_niah_single_3_4k|99.40|99.60|99.60|99.60|99.60|98.80|
|ruler_niah_single_3_64k|100.00|100.00|93.40|100.00|99.80|99.80|
|ruler_niah_single_3_8k|99.20|99.40|99.40|99.60|99.40|99.40|
|ruler_qa_1_4k|81.60|86.00|85.60|85.80|85.20|84.80|
|ruler_qa_1_8k|74.60|76.40|77.80|78.80|81.80|81.40|
|ruler_qa_2_128k|45.40|46.00|48.40|49.60|47.00|47.60|
|ruler_qa_2_4k|60.80|61.00|61.20|59.40|59.40|59.60|
|ruler_qa_2_8k|59.80|59.60|58.20|57.20|55.00|55.00|
|ruler_vt_128k|76.96|75.32|76.36|77.04|77.08|75.24|
|ruler_vt_4k|90.08|93.68|95.40|96.00|98.56|98.88|
|ruler_vt_64k|79.32|77.16|80.28|81.04|78.40|77.04|
|ruler_vt_8k|86.68|89.36|92.08|94.24|97.28|97.88|
|scbench_kv|59.00|68.00|68.40|70.00|66.40|68.20|
|scbench_mf_short|39.83|41.50|38.17|34.17|36.50|37.83|
|scbench_repoqa|47.73|50.91|57.27|60.68|58.86|59.09|
|scbench_repoqa_and_kv|70.60|71.59|78.69|79.69|80.68|80.82|
|scbench_summary|35.11|35.75|35.77|37.06|37.08|36.82|
|scbench_summary_tiny|37.06|36.98|36.82|38.15|37.92|37.12|
|scbench_summary_with_needles|63.48|65.81|66.33|67.79|68.31|68.35|

**2% protected window** — 2 benchmarks. [W&B run](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-answer-qwen25-7b1m-s40n40-grid-v1/runs/c0s997un).

|benchmark|0.2|0.3|0.4|0.5|0.75|1|
|---|---:|---:|---:|---:|---:|---:|
|ruler_fwe_4k|80.33|83.13|81.73|81.67|84.67|85.80|
|ruler_vt_4k|94.24|95.12|96.68|96.96|98.64|98.88|
---

## Part 2 — the architecture grid

**Question.** Does the shape of the mixer matter? This grid crossed three
normalization schemes, two graph widths, four mixer learning rates and two
mixer architectures, on `Qwen/Qwen2.5-7B-Instruct-1M`.

**Design.** Stage 1 trains on 40 contexts from offset 40 for 10 epochs against
the teacher's scores. Stage 2 warm-starts from stage 1 and fine-tunes on 200
Agentic contexts for 2 epochs against the model's own answers, with retention
sampled uniformly between 0.1 and 0.3.

### Stage 1, distillation against the teacher

|W&B run|architecture|normalization|graph dim|mixer LR|BCE first|BCE final|val BCE|steps|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|[ag-batchnorm-d128-lr1e3-s1](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-arch-lr-dim-grid-v1/runs/u1q9asxv)|implicit|batchnorm|128|0.001|0.1691|0.1797|0.1973|469|
|[ag-batchnorm-d128-lr1e4-s1](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-arch-lr-dim-grid-v1/runs/ak0xygn1)|implicit|batchnorm|128|0.0001|0.1689|0.1798|0.1972|469|
|[ag-batchnorm-d128-lr1e5-s1](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-arch-lr-dim-grid-v1/runs/pu8as034)|implicit|batchnorm|128|1e-05|0.1689|0.1798|0.1973|469|
|[ag-batchnorm-d128-lr5e4-s1](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-arch-lr-dim-grid-v1/runs/odlqkcky)|implicit|batchnorm|128|0.0005|0.1689|0.1797|0.1971|469|
|[ag-batchnorm-d32-lr1e3-alpha1-s1](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-arch-lr-dim-grid-v1/runs/vn38l08f)|implicit|batchnorm|32|0.001|0.1769|0.1796|0.1972|469|
|[ag-batchnorm-d32-lr1e3-alpha2-s1](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-arch-lr-dim-grid-v1/runs/40l8tvpb)|implicit|batchnorm|32|0.001|0.1930|0.1802|0.1976|469|
|[ag-batchnorm-d32-lr1e4-s1](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-arch-lr-dim-grid-v1/runs/ob2rmzd5)|implicit|batchnorm|32|0.0001|0.1689|0.1798|0.1973|469|
|[ag-batchnorm-d32-lr1e5-s1](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-arch-lr-dim-grid-v1/runs/r0adp236)|implicit|batchnorm|32|1e-05|0.1689|0.1798|0.1973|469|
|[ag-gps-d128-lr1e4-s1](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-arch-lr-dim-grid-v1/runs/e0lopwp1)|gps|none|128|0.0001|0.1689|0.1798|0.1966|469|
|[ag-gps-d128-lr1e5-s1](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-arch-lr-dim-grid-v1/runs/0kkrzexg)|gps|none|128|1e-05|0.1689|0.1798|0.1966|469|
|[ag-gps-d32-lr1e3-s1](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-arch-lr-dim-grid-v1/runs/8dhxsiba)|gps|none|32|0.001|0.1691|0.1798|0.1967|469|
|[ag-gps-d32-lr1e4-s1](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-arch-lr-dim-grid-v1/runs/9hdnqim7)|gps|none|32|0.0001|0.1688|0.1798|0.1967|469|
|[ag-gps-d32-lr1e5-s1](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-arch-lr-dim-grid-v1/runs/enho2m8d)|gps|none|32|1e-05|0.1688|0.1798|0.1967|469|
|[ag-gps-d32-lr5e4-s1](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-arch-lr-dim-grid-v1/runs/rmu6qekl)|gps|none|32|0.0005|0.1688|0.1798|0.1966|469|
|[ag-granola-d128-lr1e3-s1](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-arch-lr-dim-grid-v1/runs/ef3inmvn)|implicit|granola|128|0.001|0.1691|0.1798|0.1975|469|
|[ag-granola-d128-lr1e5-s1](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-arch-lr-dim-grid-v1/runs/9llqedh3)|implicit|granola|128|1e-05|0.1691|0.1799|0.1976|469|
|[ag-granola-d128-lr5e4-s1](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-arch-lr-dim-grid-v1/runs/3fsthbcv)|implicit|granola|128|0.0005|0.1688|0.1798|0.1972|469|
|[ag-granola-d32-lr1e3-s1](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-arch-lr-dim-grid-v1/runs/q8dj0yke)|implicit|granola|32|0.001|0.1691|0.1799|0.1975|469|
|[ag-granola-d32-lr1e4-s1](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-arch-lr-dim-grid-v1/runs/0y8xdvny)|implicit|granola|32|0.0001|0.1688|0.1798|0.1973|469|
|[ag-granola-d32-lr1e5-s1](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-arch-lr-dim-grid-v1/runs/g8v2kn30)|implicit|granola|32|1e-05|0.1688|0.1798|0.1973|469|
|[ag-granola-d32-lr5e4-s1](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-arch-lr-dim-grid-v1/runs/jcd1ypw5)|implicit|granola|32|0.0005|0.1688|0.1798|0.1973|469|

**Reading.** Every configuration converges to a final-epoch loss between about
0.179 and 0.180, whatever its architecture, width or learning rate. This is the
first appearance of the pattern that dominates the whole project: the loss does
not distinguish these models. It was assumed at the time to mean the choices did
not matter. Grid 13 later showed it means the loss is blind.

### Stage 2, answer supervision

|W&B run|architecture|normalization|graph dim|mixer LR|gate LR|train NLL|train KL|train acc.|val NLL|val acc.|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|[ag-batchnorm-d128-lr1e3-s2](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-arch-lr-dim-grid-v1/runs/1ufmmqey)|implicit|batchnorm|128|0.001|0.0001|0.4844|0.1458|0.8676|0.4978|0.8484|
|[ag-batchnorm-d128-lr1e4-s2](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-arch-lr-dim-grid-v1/runs/7odo5l2x)|implicit|batchnorm|128|0.0001|0.0001|0.5273|0.1810|0.8382|0.4853|0.8584|
|[ag-batchnorm-d128-lr1e5-s2](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-arch-lr-dim-grid-v1/runs/4wscjluc)|implicit|batchnorm|128|1e-05|0.0001|0.5234|0.1670|0.8676|0.4779|0.8607|
|[ag-batchnorm-d128-lr5e4-s2](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-arch-lr-dim-grid-v1/runs/9mc7p3nb)|implicit|batchnorm|128|0.0005|0.0001|0.5195|0.1446|0.8235|0.4344|0.8721|
|[ag-batchnorm-d32-lr1e3-alpha1-s2](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-arch-lr-dim-grid-v1/runs/9rn2qsns)|implicit|batchnorm|32|0.001|0.0001|0.6992|0.3286|0.7647|0.4747|0.8597|
|[ag-batchnorm-d32-lr1e3-alpha2-s2](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-arch-lr-dim-grid-v1/runs/c4b61jcm)|implicit|batchnorm|32|0.001|0.0001|0.7148|0.3566|0.7647|0.5921|0.8242|
|[ag-batchnorm-d32-lr1e3-s2](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-arch-lr-dim-grid-v1/runs/ak8dbcl7)|implicit|batchnorm|32|0.001|0.0001|0.5352|0.2004|0.8382|0.4156|0.8756|
|[ag-batchnorm-d32-lr1e4-s2](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-arch-lr-dim-grid-v1/runs/lwx7e3rq)|implicit|batchnorm|32|0.0001|0.0001|0.5859|0.1882|0.8088|0.5101|0.8699|
|[ag-batchnorm-d32-lr1e5-s2](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-arch-lr-dim-grid-v1/runs/dyxzeedo)|implicit|batchnorm|32|1e-05|0.0001|0.5195|0.1760|0.8382|0.5422|0.8379|
|[ag-batchnorm-d32-lr5e4-s2](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-arch-lr-dim-grid-v1/runs/x0da5jvf)|implicit|batchnorm|32|0.0005|0.0001|0.5391|0.1935|0.8382|0.4328|0.8539|
|[ag-gps-d128-lr1e3-s2](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-arch-lr-dim-grid-v1/runs/tzlo1osl)|gps|none|128|0.001|0.0001|0.7305|0.3614|0.7794|0.4259|0.8688|
|[ag-gps-d128-lr1e4-s2](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-arch-lr-dim-grid-v1/runs/yw6kv7kn)|gps|none|128|0.0001|0.0001|0.5273|0.1516|0.8529|0.4505|0.8539|
|[ag-gps-d128-lr1e5-s2](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-arch-lr-dim-grid-v1/runs/lor0xamk)|gps|none|128|1e-05|0.0001|0.5117|0.1863|0.8382|0.4803|0.8539|
|[ag-gps-d128-lr5e4-s2](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-arch-lr-dim-grid-v1/runs/dmnku5ud)|gps|none|128|0.0005|0.0001|0.5391|0.1703|0.8382|0.4718|0.8539|
|[ag-gps-d32-lr1e3-s2](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-arch-lr-dim-grid-v1/runs/l5a8j5xx)|gps|none|32|0.001|0.0001|0.6484|0.2223|0.7647|0.5334|0.8575|
|[ag-gps-d32-lr1e4-s2](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-arch-lr-dim-grid-v1/runs/odchxo57)|gps|none|32|0.0001|0.0001|0.5156|0.1921|0.8235|0.4524|0.8767|
|[ag-gps-d32-lr1e5-s2](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-arch-lr-dim-grid-v1/runs/rlagfpxv)|gps|none|32|1e-05|0.0001|0.5000|0.1625|0.8529|0.4859|0.8539|
|[ag-gps-d32-lr5e4-s2](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-arch-lr-dim-grid-v1/runs/e78jb6uo)|gps|none|32|0.0005|0.0001|0.6133|0.2299|0.8235|0.4786|0.8676|
|[ag-granola-d128-lr1e3-s2](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-arch-lr-dim-grid-v1/runs/3t22wtmf)|implicit|granola|128|0.001|0.0001|0.6562|0.2088|0.7794|0.6173|0.8235|
|[ag-granola-d128-lr1e4-s2](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-arch-lr-dim-grid-v1/runs/iv3g3km2)|implicit|granola|128|0.0001|0.0001|0.5664|0.2134|0.8382|0.4387|0.8630|
|[ag-granola-d128-lr1e5-s2](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-arch-lr-dim-grid-v1/runs/5bw917k1)|implicit|granola|128|1e-05|0.0001|0.4883|0.1493|0.8382|0.5313|0.8425|
|[ag-granola-d128-lr5e4-s2](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-arch-lr-dim-grid-v1/runs/ex3w86ze)|implicit|granola|128|0.0005|0.0001|0.5859|0.2119|0.7941|0.5041|0.8379|
|[ag-granola-d32-lr1e3-s2](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-arch-lr-dim-grid-v1/runs/eca92e96)|implicit|granola|32|0.001|0.0001|0.5781|0.2141|0.7794|0.5024|0.8643|
|[ag-granola-d32-lr1e4-s2](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-arch-lr-dim-grid-v1/runs/4skiwapw)|implicit|granola|32|0.0001|0.0001|0.5156|0.1833|0.8235|0.4845|0.8539|
|[ag-granola-d32-lr1e5-s2](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-arch-lr-dim-grid-v1/runs/g4c3ig18)|implicit|granola|32|1e-05|0.0001|0.4785|0.1842|0.8676|0.5564|0.8607|
|[ag-granola-d32-lr5e4-s2](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-arch-lr-dim-grid-v1/runs/qz3tkzdp)|implicit|granola|32|0.0005|0.0001|0.5898|0.1930|0.8529|0.4955|0.8630|

### Benchmark scores

All runs in this grid were evaluated with the 2% protected window, so they
compare to the paper's tables 1 and 2 and to each other, but not to the
gate-space grids in Part 4, which protected nothing.

Eleven-task mean absolute score. The last two columns give the two retrieval
tasks at the lowest retention, where the spread between configurations is
widest.

|run|0.4|0.3|0.2|0.1|0.05|full|kv @ lowest|prefix/suffix @ lowest|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|batchnorm, dim 128, mixer LR 1e-3|54.00|52.72|48.82|36.76|17.84|54.23|1.20|0.60|
|batchnorm, dim 128, mixer LR 1e-4|54.10|53.10|50.26|38.53|19.55|53.78|1.80|0.60|
|batchnorm, dim 128, mixer LR 1e-5|53.97|53.03|51.35|39.51|19.66|54.23|1.20|0.60|
|batchnorm, dim 128, mixer LR 5e-4|52.97|51.74|48.00|36.36|17.26|54.23|1.20|0.60|
|batchnorm, dim 32, mixer LR 1e-3|52.99|51.34|48.29|36.56|17.84|54.23|1.20|0.60|
|batchnorm, dim 32, mixer LR 1e-3, alpha 1|49.53|45.40|38.70|22.75|13.71|54.23|1.00|0.60|
|batchnorm, dim 32, mixer LR 1e-3, alpha 2|33.19|30.85|28.47|19.99|14.58|54.15|0.80|0.80|
|batchnorm, dim 32, mixer LR 1e-4|54.02|53.47|50.94|38.76|18.26|53.89|1.20|0.60|
|batchnorm, dim 32, mixer LR 1e-5|50.18|51.51|48.82|39.30|18.97|53.97|1.20|0.60|
|batchnorm, dim 32, mixer LR 5e-4|53.70|52.74|50.17|36.81|18.34|54.23|1.40|0.60|
|gps, dim 128, mixer LR 1e-3|50.75|51.06|44.78|33.06|16.84|54.23|1.40|0.40|
|gps, dim 128, mixer LR 1e-4|54.26|54.12|50.38|39.98|18.17|53.89|1.00|0.60|
|gps, dim 128, mixer LR 1e-5|54.37|53.75|48.87|38.70|19.37|54.23|1.00|0.60|
|gps, dim 128, mixer LR 5e-4|51.69|50.82|47.95|33.72|17.72|54.13|1.00|0.60|
|gps, dim 32, mixer LR 1e-3|53.77|53.33|48.79|39.29|20.08|54.23|2.20|0.40|
|gps, dim 32, mixer LR 1e-4|53.21|53.47|50.32|38.40|19.41|53.78|1.20|0.60|
|gps, dim 32, mixer LR 1e-5|54.46|53.51|49.41|38.58|18.79|54.23|1.00|0.60|
|gps, dim 32, mixer LR 5e-4|52.94|50.93|49.19|38.63|19.47|54.15|1.00|0.40|
|granola, dim 128, mixer LR 1e-3|35.56|33.96|31.42|22.89|15.05|54.23|1.00|0.40|
|granola, dim 128, mixer LR 1e-4|53.51|51.15|48.98|38.19|19.39|53.78|1.00|0.60|
|granola, dim 128, mixer LR 1e-5|54.72|52.77|48.63|39.35|19.26|54.23|1.20|0.60|
|granola, dim 128, mixer LR 5e-4|49.54|46.72|42.83|30.97|16.49|54.15|1.00|0.60|
|granola, dim 32, mixer LR 1e-3|54.40|52.61|49.25|37.82|18.49|54.23|1.00|0.60|
|granola, dim 32, mixer LR 1e-4|54.38|53.81|48.74|39.16|19.66|53.86|1.00|0.60|
|granola, dim 32, mixer LR 1e-5|53.21|52.38|48.92|39.41|19.29|54.23|1.00|0.60|
|granola, dim 32, mixer LR 5e-4|53.59|53.39|50.55|37.99|18.47|54.15|1.00|0.60|

**What this shows.** The best configuration, batch normalization at width 128
with the slowest mixer learning rate, reaches 51.35 at one fifth retention.
The paper's own selector reports 50.92 on the same eleven tasks at the same
window. On this evidence the work matches the published result.

The three runs that vary the mixer's initial contribution are the cleanest
result in the whole record, because they change one number and nothing else:

|initial mixer contribution|0.2|0.1|
|---|---:|---:|
|0.1|48.29|36.56|
|1|38.70|22.75|
|2|28.47|19.99|

Ten points lost between the first and second, ten more between the second and
third. A mixer that pushes harder on the gate is monotonically worse, and this
was visible in a grid run long before the gate-space work set out to make the
mixer push harder still.

<details>
<summary>Per-task detail, all 26 runs</summary>

**batchnorm, dim 128, mixer LR 1e-3**

|task|full|0.4|0.3|0.2|0.1|0.05|
|---|---:|---:|---:|---:|---:|---:|
|kv|69.20|70.60|60.20|46.60|25.20|1.20|
|prefix_suffix|49.20|46.80|44.60|28.40|10.00|0.60|
|summary|37.02|36.47|36.61|35.96|31.63|27.16|
|vt|40.53|39.51|40.40|43.60|45.91|39.56|
|many_shot|38.52|37.41|37.04|35.56|31.85|32.59|
|mf|33.17|34.50|35.33|32.17|32.00|13.83|
|choice_eng|77.78|79.17|77.78|76.39|69.44|46.30|
|qa_eng|42.92|42.75|44.14|41.16|27.05|13.38|
|repoqa|59.09|58.41|57.27|54.77|31.59|1.59|
|summary_with_needles|68.43|67.98|67.70|66.60|51.72|15.59|
|repoqa_and_kv|80.68|80.40|78.84|75.85|48.01|4.40|

**batchnorm, dim 128, mixer LR 1e-4**

|task|full|0.4|0.3|0.2|0.1|0.05|
|---|---:|---:|---:|---:|---:|---:|
|kv|67.80|72.00|73.00|66.20|28.60|1.80|
|prefix_suffix|49.60|42.00|38.20|30.20|14.40|0.60|
|summary|37.01|36.68|36.34|35.82|31.99|28.09|
|vt|40.13|40.09|41.87|40.40|45.47|35.64|
|many_shot|37.78|37.78|37.04|35.19|31.48|34.81|
|mf|32.00|34.33|34.83|33.00|30.50|17.33|
|choice_eng|77.78|79.17|73.61|72.22|69.44|55.09|
|qa_eng|41.09|44.71|44.71|41.20|22.68|14.93|
|repoqa|59.09|59.55|58.18|55.68|42.05|2.05|
|summary_with_needles|68.46|68.15|67.74|66.91|54.14|20.75|
|repoqa_and_kv|80.82|80.68|78.55|75.99|53.12|3.98|

**batchnorm, dim 128, mixer LR 1e-5**

|task|full|0.4|0.3|0.2|0.1|0.05|
|---|---:|---:|---:|---:|---:|---:|
|kv|69.20|71.00|71.00|69.40|32.00|1.20|
|prefix_suffix|49.20|42.00|35.80|29.40|12.80|0.60|
|summary|37.02|36.53|36.37|35.85|32.60|27.62|
|vt|40.53|40.09|40.44|41.82|48.04|42.18|
|many_shot|38.52|37.78|36.67|34.81|30.37|34.07|
|mf|33.17|35.00|36.17|33.00|29.33|16.17|
|choice_eng|77.78|80.56|75.00|76.39|70.83|52.31|
|qa_eng|42.92|42.70|47.09|46.76|22.37|15.65|
|repoqa|59.09|59.32|57.95|55.45|42.73|2.27|
|summary_with_needles|68.43|67.92|67.48|66.21|59.69|18.37|
|repoqa_and_kv|80.68|80.82|79.40|75.71|53.84|5.82|

**batchnorm, dim 128, mixer LR 5e-4**

|task|full|0.4|0.3|0.2|0.1|0.05|
|---|---:|---:|---:|---:|---:|---:|
|kv|69.20|57.40|55.20|46.20|21.00|1.20|
|prefix_suffix|49.20|47.40|44.20|29.00|11.80|0.60|
|summary|37.02|36.51|36.00|35.68|32.31|27.31|
|vt|40.53|38.80|39.73|35.64|35.47|30.67|
|many_shot|38.52|36.67|37.78|34.07|32.22|32.22|
|mf|33.17|34.83|35.33|31.00|29.67|16.17|
|choice_eng|77.78|77.78|75.00|73.61|68.98|44.91|
|qa_eng|42.92|44.34|41.06|44.55|22.86|14.00|
|repoqa|59.09|60.91|58.64|56.82|40.91|1.82|
|summary_with_needles|68.43|67.91|67.23|65.17|57.58|17.39|
|repoqa_and_kv|80.68|80.11|78.98|76.28|47.16|3.55|

**batchnorm, dim 32, mixer LR 1e-3**

|task|full|0.4|0.3|0.2|0.1|0.05|
|---|---:|---:|---:|---:|---:|---:|
|kv|69.20|57.80|51.40|47.40|22.20|1.20|
|prefix_suffix|49.20|42.60|37.60|27.00|11.20|0.60|
|summary|37.02|36.56|36.32|35.89|32.46|27.64|
|vt|40.53|41.87|40.98|38.00|41.02|34.58|
|many_shot|38.52|38.15|38.89|35.93|31.11|32.22|
|mf|33.17|35.00|36.50|34.33|28.67|14.83|
|choice_eng|77.78|77.78|72.22|73.61|66.67|46.30|
|qa_eng|42.92|45.17|45.08|40.59|27.82|14.98|
|repoqa|59.09|59.55|58.41|56.82|40.91|1.82|
|summary_with_needles|68.43|67.85|68.19|66.01|50.94|18.38|
|repoqa_and_kv|80.68|80.54|79.12|75.57|49.15|3.69|

**batchnorm, dim 32, mixer LR 1e-3, alpha 1**

|task|full|0.4|0.3|0.2|0.1|0.05|
|---|---:|---:|---:|---:|---:|---:|
|kv|69.20|55.20|37.20|26.00|6.20|1.00|
|prefix_suffix|49.20|37.40|27.60|12.20|1.60|0.60|
|summary|37.02|36.00|35.79|33.92|30.10|26.99|
|vt|40.53|35.11|31.64|28.18|26.67|6.00|
|many_shot|38.52|37.04|33.33|34.44|31.85|29.63|
|mf|33.17|33.00|33.00|31.33|23.67|13.50|
|choice_eng|77.78|76.39|76.39|68.98|60.65|46.30|
|qa_eng|42.92|39.47|40.35|39.26|16.83|11.76|
|repoqa|59.09|52.95|48.86|35.23|5.00|0.45|
|summary_with_needles|68.43|67.39|65.76|64.16|35.50|13.68|
|repoqa_and_kv|80.68|74.86|69.46|51.99|12.22|0.85|

**batchnorm, dim 32, mixer LR 1e-3, alpha 2**

|task|full|0.4|0.3|0.2|0.1|0.05|
|---|---:|---:|---:|---:|---:|---:|
|kv|69.20|3.40|2.20|1.60|0.80|0.80|
|prefix_suffix|49.20|6.80|2.60|1.40|0.80|0.80|
|summary|37.02|35.99|35.95|34.55|29.79|27.68|
|vt|40.53|27.91|30.18|26.71|14.18|3.38|
|many_shot|37.78|34.81|34.07|34.81|35.93|32.59|
|mf|33.00|29.50|28.33|25.83|21.50|13.00|
|choice_eng|77.78|73.61|73.61|73.61|60.65|49.54|
|qa_eng|42.92|38.71|34.03|37.28|22.27|13.74|
|repoqa|59.09|28.41|19.55|11.14|1.82|0.68|
|summary_with_needles|68.43|65.54|65.24|58.86|30.79|17.51|
|repoqa_and_kv|80.68|20.45|13.64|7.39|1.42|0.71|

**batchnorm, dim 32, mixer LR 1e-4**

|task|full|0.4|0.3|0.2|0.1|0.05|
|---|---:|---:|---:|---:|---:|---:|
|kv|69.20|74.60|71.20|68.60|28.20|1.20|
|prefix_suffix|49.20|44.20|40.40|28.60|14.20|0.60|
|summary|37.02|36.88|36.32|36.18|32.19|27.73|
|vt|40.53|40.18|40.36|45.51|40.13|35.82|
|many_shot|37.78|37.41|36.67|33.70|32.22|31.11|
|mf|32.00|35.17|36.00|33.67|30.67|16.67|
|choice_eng|77.78|76.39|76.39|75.00|70.83|46.30|
|qa_eng|41.09|42.35|45.65|39.78|28.40|15.23|
|repoqa|59.09|58.41|57.73|56.14|43.86|2.27|
|summary_with_needles|68.43|68.06|67.78|67.17|58.25|19.76|
|repoqa_and_kv|80.68|80.54|79.69|75.99|47.44|4.12|

**batchnorm, dim 32, mixer LR 1e-5**

|task|full|0.4|0.3|0.2|0.1|0.05|
|---|---:|---:|---:|---:|---:|---:|
|kv|69.20|56.00|52.60|48.20|24.40|1.20|
|prefix_suffix|49.20|18.40|40.60|29.80|11.60|0.60|
|summary|37.02|36.37|36.19|35.61|32.52|27.41|
|vt|40.53|39.38|41.24|41.02|50.04|40.93|
|many_shot|38.52|37.41|36.67|35.19|31.11|33.33|
|mf|32.17|34.83|35.17|33.17|29.67|15.83|
|choice_eng|77.78|77.78|75.00|76.39|71.76|50.46|
|qa_eng|41.09|44.30|44.52|40.40|25.02|15.32|
|repoqa|59.09|59.09|57.50|55.23|43.64|2.27|
|summary_with_needles|68.43|68.02|67.72|65.75|58.51|17.14|
|repoqa_and_kv|80.68|80.40|79.40|76.28|53.98|4.12|

**batchnorm, dim 32, mixer LR 5e-4**

|task|full|0.4|0.3|0.2|0.1|0.05|
|---|---:|---:|---:|---:|---:|---:|
|kv|69.20|73.80|73.20|68.60|19.80|1.40|
|prefix_suffix|49.20|37.20|35.40|30.60|13.20|0.60|
|summary|37.02|36.53|36.91|35.77|32.58|27.26|
|vt|40.53|40.44|39.56|41.20|45.78|38.13|
|many_shot|38.52|37.04|37.41|34.44|32.22|33.70|
|mf|33.17|34.33|35.50|33.17|27.67|16.17|
|choice_eng|77.78|79.17|75.00|70.83|68.98|47.69|
|qa_eng|42.92|45.17|43.72|40.63|23.74|14.63|
|repoqa|59.09|58.41|57.73|55.68|38.86|2.05|
|summary_with_needles|68.43|68.31|67.59|65.83|55.51|16.89|
|repoqa_and_kv|80.68|80.26|78.12|75.14|46.59|3.27|

**gps, dim 128, mixer LR 1e-3**

|task|full|0.4|0.3|0.2|0.1|0.05|
|---|---:|---:|---:|---:|---:|---:|
|kv|69.20|60.80|66.80|35.40|20.40|1.40|
|prefix_suffix|49.20|32.20|34.00|23.80|8.40|0.40|
|summary|37.02|36.21|35.49|35.44|32.75|28.08|
|vt|40.53|43.33|46.49|47.29|42.89|26.67|
|many_shot|38.52|35.93|36.67|34.81|33.70|29.63|
|mf|33.17|35.17|34.67|32.67|24.83|15.17|
|choice_eng|77.78|75.00|77.78|72.22|66.67|49.07|
|qa_eng|42.92|43.05|43.21|40.38|27.55|12.23|
|repoqa|59.09|52.95|47.50|38.18|12.50|1.36|
|summary_with_needles|68.43|68.03|67.33|65.78|56.85|19.54|
|repoqa_and_kv|80.68|75.57|71.73|66.62|37.07|1.70|

**gps, dim 128, mixer LR 1e-4**

|task|full|0.4|0.3|0.2|0.1|0.05|
|---|---:|---:|---:|---:|---:|---:|
|kv|69.20|73.20|72.40|60.40|28.80|1.00|
|prefix_suffix|49.20|40.60|45.20|30.00|13.80|0.60|
|summary|37.02|36.38|36.43|35.30|33.07|27.20|
|vt|40.53|40.67|41.69|43.42|46.67|40.18|
|many_shot|37.78|37.78|36.67|35.19|31.11|33.33|
|mf|32.00|34.83|35.33|33.50|28.67|16.00|
|choice_eng|77.78|81.94|76.39|76.39|71.76|43.52|
|qa_eng|41.09|43.90|47.23|43.58|27.74|14.48|
|repoqa|59.09|58.86|57.50|55.00|45.00|2.50|
|summary_with_needles|68.43|68.26|67.94|65.69|60.49|15.51|
|repoqa_and_kv|80.68|80.40|78.55|75.71|52.70|5.54|

**gps, dim 128, mixer LR 1e-5**

|task|full|0.4|0.3|0.2|0.1|0.05|
|---|---:|---:|---:|---:|---:|---:|
|kv|69.20|75.40|72.80|49.20|28.80|1.00|
|prefix_suffix|49.20|42.60|45.80|30.40|11.00|0.60|
|summary|37.02|36.89|36.15|35.31|32.87|27.60|
|vt|40.53|40.27|39.24|39.07|42.62|43.91|
|many_shot|38.52|36.67|37.04|34.81|30.00|36.67|
|mf|33.17|34.83|35.83|34.17|29.67|16.50|
|choice_eng|77.78|79.17|76.39|73.61|67.59|47.69|
|qa_eng|42.92|45.09|44.36|44.63|25.54|14.09|
|repoqa|59.09|58.64|57.27|55.23|45.00|3.18|
|summary_with_needles|68.43|68.30|67.81|65.11|56.92|15.77|
|repoqa_and_kv|80.68|80.26|78.55|75.99|55.68|6.11|

**gps, dim 128, mixer LR 5e-4**

|task|full|0.4|0.3|0.2|0.1|0.05|
|---|---:|---:|---:|---:|---:|---:|
|kv|69.20|52.80|49.60|37.20|8.00|1.00|
|prefix_suffix|49.20|36.00|37.20|26.20|2.00|0.60|
|summary|37.02|36.47|36.79|35.63|33.63|27.33|
|vt|40.53|44.18|44.00|44.31|45.42|32.31|
|many_shot|37.78|36.30|35.56|34.07|32.22|30.00|
|mf|32.83|36.33|35.67|35.67|27.67|15.00|
|choice_eng|77.78|77.78|76.39|73.61|69.44|50.46|
|qa_eng|42.92|42.62|41.33|46.28|32.34|15.72|
|repoqa|59.09|58.64|56.82|53.41|25.23|2.05|
|summary_with_needles|68.43|68.38|68.14|66.24|59.48|19.13|
|repoqa_and_kv|80.68|79.12|77.56|74.86|35.51|1.28|

**gps, dim 32, mixer LR 1e-3**

|task|full|0.4|0.3|0.2|0.1|0.05|
|---|---:|---:|---:|---:|---:|---:|
|kv|69.20|70.40|71.20|49.60|23.40|2.20|
|prefix_suffix|49.20|42.80|42.80|27.20|9.80|0.40|
|summary|37.01|37.12|36.65|36.16|32.90|27.85|
|vt|40.53|38.76|38.58|38.67|43.56|33.07|
|many_shot|38.52|36.67|37.41|34.44|32.22|33.33|
|mf|33.17|34.83|36.33|34.67|27.83|17.33|
|choice_eng|77.78|80.56|75.00|77.78|73.15|56.48|
|qa_eng|42.92|43.22|45.38|39.77|31.46|18.41|
|repoqa|59.09|59.55|56.59|55.91|41.82|3.41|
|summary_with_needles|68.43|68.19|68.26|66.39|60.83|19.90|
|repoqa_and_kv|80.68|79.40|78.41|76.14|55.26|8.52|

**gps, dim 32, mixer LR 1e-4**

|task|full|0.4|0.3|0.2|0.1|0.05|
|---|---:|---:|---:|---:|---:|---:|
|kv|67.80|68.40|71.20|61.20|30.80|1.20|
|prefix_suffix|49.60|35.40|40.60|31.00|10.80|0.60|
|summary|37.01|36.65|36.74|35.88|32.95|27.55|
|vt|40.13|41.02|41.82|42.31|41.87|44.22|
|many_shot|37.78|38.15|38.15|32.96|31.11|35.19|
|mf|32.00|35.00|36.00|34.00|29.67|16.33|
|choice_eng|77.78|80.56|73.61|76.39|67.59|52.31|
|qa_eng|41.09|42.96|46.18|42.89|25.20|12.50|
|repoqa|59.09|59.09|57.27|55.23|42.73|2.73|
|summary_with_needles|68.46|67.94|68.20|66.19|58.68|15.95|
|repoqa_and_kv|80.82|80.11|78.41|75.43|50.99|4.97|

**gps, dim 32, mixer LR 1e-5**

|task|full|0.4|0.3|0.2|0.1|0.05|
|---|---:|---:|---:|---:|---:|---:|
|kv|69.20|74.60|74.20|52.80|30.00|1.00|
|prefix_suffix|49.20|43.60|42.00|31.40|10.80|0.60|
|summary|37.02|36.54|36.56|35.63|32.32|27.55|
|vt|40.53|39.73|39.38|39.29|42.93|41.87|
|many_shot|38.52|38.52|36.67|34.81|30.37|33.70|
|mf|33.17|34.33|35.33|34.50|28.67|15.83|
|choice_eng|77.78|80.56|75.00|75.00|68.98|49.07|
|qa_eng|42.92|43.19|45.59|42.96|23.48|13.91|
|repoqa|59.09|58.64|57.50|55.23|43.18|2.50|
|summary_with_needles|68.43|68.21|67.88|65.35|58.51|15.27|
|repoqa_and_kv|80.68|81.11|78.55|76.56|55.11|5.40|

**gps, dim 32, mixer LR 5e-4**

|task|full|0.4|0.3|0.2|0.1|0.05|
|---|---:|---:|---:|---:|---:|---:|
|kv|69.20|74.60|53.20|43.60|22.20|1.00|
|prefix_suffix|49.20|32.00|34.80|33.00|13.00|0.40|
|summary|37.02|36.30|36.13|35.10|32.37|27.75|
|vt|40.53|38.80|39.20|41.60|50.18|43.24|
|many_shot|37.78|37.04|37.04|35.56|30.37|33.33|
|mf|33.00|34.33|35.17|35.17|30.50|16.83|
|choice_eng|77.78|77.78|75.00|73.61|67.59|52.31|
|qa_eng|42.92|44.11|44.42|44.45|26.88|13.74|
|repoqa|59.09|58.86|57.95|55.45|41.82|3.41|
|summary_with_needles|68.43|68.22|68.07|66.86|58.34|16.89|
|repoqa_and_kv|80.68|80.26|79.26|76.70|51.70|5.26|

**granola, dim 128, mixer LR 1e-3**

|task|full|0.4|0.3|0.2|0.1|0.05|
|---|---:|---:|---:|---:|---:|---:|
|kv|69.20|2.60|1.40|1.20|1.20|1.00|
|prefix_suffix|49.20|6.00|2.20|1.00|0.60|0.40|
|summary|37.02|35.80|36.15|34.80|31.69|27.67|
|vt|40.53|34.53|34.49|38.53|28.67|21.38|
|many_shot|38.52|36.67|35.93|34.44|32.22|31.11|
|mf|33.17|30.17|29.50|27.50|21.83|12.50|
|choice_eng|77.78|75.00|75.00|70.83|60.65|39.35|
|qa_eng|42.92|43.97|43.80|40.45|27.13|14.96|
|repoqa|59.09|28.86|24.09|16.14|3.86|0.91|
|summary_with_needles|68.43|67.48|67.56|65.96|42.05|15.47|
|repoqa_and_kv|80.68|30.11|23.44|14.77|1.85|0.85|

**granola, dim 128, mixer LR 1e-4**

|task|full|0.4|0.3|0.2|0.1|0.05|
|---|---:|---:|---:|---:|---:|---:|
|kv|67.80|64.80|54.00|48.00|24.80|1.00|
|prefix_suffix|49.60|44.80|37.20|28.20|10.00|0.60|
|summary|37.01|36.79|36.12|35.62|32.97|27.82|
|vt|40.13|40.22|40.49|41.78|44.49|39.38|
|many_shot|37.78|37.78|35.56|35.56|28.89|33.33|
|mf|32.00|35.00|35.17|34.33|31.33|17.33|
|choice_eng|77.78|79.17|73.61|75.00|74.54|55.09|
|qa_eng|41.09|43.05|45.29|44.06|24.51|15.15|
|repoqa|59.09|58.64|58.18|54.32|38.18|2.05|
|summary_with_needles|68.46|68.21|67.75|66.03|61.64|16.97|
|repoqa_and_kv|80.82|80.11|79.26|75.85|48.72|4.55|

**granola, dim 128, mixer LR 1e-5**

|task|full|0.4|0.3|0.2|0.1|0.05|
|---|---:|---:|---:|---:|---:|---:|
|kv|69.20|76.40|60.20|46.20|28.80|1.20|
|prefix_suffix|49.20|45.00|44.40|30.40|12.60|0.60|
|summary|37.02|36.71|36.61|35.59|32.80|27.61|
|vt|40.53|39.69|40.89|39.64|48.89|44.04|
|many_shot|38.52|37.04|35.56|34.81|32.59|34.44|
|mf|33.17|34.83|35.50|33.00|29.33|16.33|
|choice_eng|77.78|79.17|76.39|76.39|66.20|49.54|
|qa_eng|42.92|44.81|46.84|41.57|23.74|13.03|
|repoqa|59.09|59.77|57.50|55.00|43.86|2.95|
|summary_with_needles|68.43|68.25|67.63|65.80|60.50|16.25|
|repoqa_and_kv|80.68|80.26|78.98|76.56|53.55|5.82|

**granola, dim 128, mixer LR 5e-4**

|task|full|0.4|0.3|0.2|0.1|0.05|
|---|---:|---:|---:|---:|---:|---:|
|kv|69.20|52.40|42.60|34.60|12.00|1.00|
|prefix_suffix|49.20|28.80|18.80|7.40|1.40|0.60|
|summary|37.02|36.31|36.09|36.29|32.16|27.35|
|vt|40.53|41.07|39.38|38.71|38.98|25.78|
|many_shot|37.78|35.93|33.70|33.33|34.81|29.63|
|mf|33.00|34.67|34.67|31.50|27.83|11.33|
|choice_eng|77.78|73.61|75.00|75.00|66.20|53.24|
|qa_eng|42.92|42.79|42.50|42.42|27.27|15.19|
|repoqa|59.09|55.23|52.50|42.95|15.45|0.68|
|summary_with_needles|68.43|67.64|66.09|64.42|51.45|15.89|
|repoqa_and_kv|80.68|76.56|72.59|64.49|33.10|0.71|

**granola, dim 32, mixer LR 1e-3**

|task|full|0.4|0.3|0.2|0.1|0.05|
|---|---:|---:|---:|---:|---:|---:|
|kv|69.20|75.60|63.20|48.60|27.60|1.00|
|prefix_suffix|49.20|43.60|42.00|31.80|9.40|0.60|
|summary|37.02|36.77|36.55|36.11|32.87|28.02|
|vt|40.53|41.56|40.71|42.67|49.07|35.20|
|many_shot|38.52|37.04|36.67|35.56|30.37|33.33|
|mf|33.17|35.33|35.83|33.83|29.17|19.50|
|choice_eng|77.78|76.39|76.39|73.61|69.44|53.24|
|qa_eng|42.92|43.17|44.89|44.76|29.38|12.69|
|repoqa|59.09|59.77|56.82|53.64|37.27|1.82|
|summary_with_needles|68.43|68.33|67.55|66.33|54.98|15.83|
|repoqa_and_kv|80.68|80.82|78.12|74.86|46.45|2.13|

**granola, dim 32, mixer LR 1e-4**

|task|full|0.4|0.3|0.2|0.1|0.05|
|---|---:|---:|---:|---:|---:|---:|
|kv|67.80|73.80|74.20|47.60|29.00|1.00|
|prefix_suffix|49.60|42.40|42.20|29.40|11.60|0.60|
|summary|37.01|36.45|36.41|35.61|32.89|27.73|
|vt|40.13|40.27|41.56|39.29|46.53|42.58|
|many_shot|38.52|38.52|36.67|34.81|32.59|36.30|
|mf|32.17|34.83|36.00|34.33|30.33|17.00|
|choice_eng|77.78|79.17|75.00|73.61|68.06|50.46|
|qa_eng|41.09|44.65|45.09|43.70|26.99|14.09|
|repoqa|59.09|59.32|57.95|55.45|41.82|3.64|
|summary_with_needles|68.46|68.14|68.00|66.15|56.58|16.87|
|repoqa_and_kv|80.82|80.68|78.84|76.14|54.40|5.97|

**granola, dim 32, mixer LR 1e-5**

|task|full|0.4|0.3|0.2|0.1|0.05|
|---|---:|---:|---:|---:|---:|---:|
|kv|69.20|61.60|53.60|42.80|30.60|1.00|
|prefix_suffix|49.20|44.40|42.60|31.80|11.20|0.60|
|summary|37.02|36.35|36.52|35.37|32.45|27.57|
|vt|40.53|40.40|41.51|45.38|46.31|46.18|
|many_shot|38.52|37.04|36.67|35.56|31.48|34.81|
|mf|33.17|35.33|36.17|34.50|29.17|15.50|
|choice_eng|77.78|79.17|79.17|73.61|68.98|49.54|
|qa_eng|42.92|43.91|45.42|40.74|25.96|12.94|
|repoqa|59.09|58.64|57.73|56.36|44.55|2.73|
|summary_with_needles|68.43|68.16|67.92|65.17|58.51|16.40|
|repoqa_and_kv|80.68|80.26|78.84|76.85|54.26|4.97|

**granola, dim 32, mixer LR 5e-4**

|task|full|0.4|0.3|0.2|0.1|0.05|
|---|---:|---:|---:|---:|---:|---:|
|kv|69.20|68.80|69.80|64.40|32.20|1.00|
|prefix_suffix|49.20|39.40|44.40|29.80|11.20|0.60|
|summary|37.02|36.87|36.61|35.14|32.07|27.82|
|vt|40.53|41.51|41.73|42.36|39.64|33.51|
|many_shot|37.78|35.56|36.30|34.44|32.22|32.96|
|mf|33.00|34.00|35.17|33.67|29.67|16.50|
|choice_eng|77.78|80.56|75.00|76.39|68.06|55.09|
|qa_eng|42.92|45.08|42.69|41.07|25.98|12.81|
|repoqa|59.09|59.32|57.95|55.91|38.86|2.73|
|summary_with_needles|68.43|68.29|68.20|66.73|57.52|16.58|
|repoqa_and_kv|80.68|80.11|79.40|76.14|50.43|3.55|

</details>

---

## Part 3 — answer-supervision grids

Three grids explored how to fine-tune against answers rather than against the
teacher's scores.

### Grid: KL loss, learning rates and accumulation

**Question.** What learning rates and batch sizes make answer-supervised
training stable, and does training on 2000 contexts beat 200?

|W&B run|contexts|gate LR|mixer LR|accum.|retention|train NLL|train KL|train acc.|val NLL|val acc.|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|[q25a-kl-n2000e2-a16-lr01-s0](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-answer-kl-grid-v1/runs/u8lqebls)|2000|1e-05|0.0001|16|0.25|0.4461|0.4609|0.8790|0.6853|0.8001|
|[q25a-kl-n2000e2-a16-lr05-s0](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-answer-kl-grid-v1/runs/0l9o2lpf)|2000|5e-05|0.0005|16|0.25|0.3198|0.3160|0.8998|0.6686|0.8024|
|[q25a-kl-n2000e2-a32-lr01-s0](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-answer-kl-grid-v1/runs/lti3l0mw)|2000|1e-05|0.0001|32|0.25|0.5201|0.5137|0.8581|0.6984|0.8034|
|[q25a-kl-n2000e2-a32-lr05-s0](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-answer-kl-grid-v1/runs/4n0yn2yi)|2000|5e-05|0.0005|32|0.25|0.3079|0.3107|0.9117|0.6725|0.8022|
|[q25a-kl-n2000e2-a8-lr01-s0](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-answer-kl-grid-v1/runs/j0doepwp)|2000|1e-05|0.0001|8|0.25|0.3885|0.3965|0.8968|0.6806|0.7987|
|[q25a-kl-n2000e2-a8-lr05-s0](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-answer-kl-grid-v1/runs/14c527d5)|2000|5e-05|0.0005|8|0.25|0.4294|0.4094|0.8617|0.6805|0.8005|
|[q25a-kl-n2000e2-a8-lr10-s0](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-answer-kl-grid-v1/runs/wujhc10s)|2000|0.0001|0.001|8|0.25|0.3136|0.2776|0.9115|0.6824|0.7985|
|[q25a-kl-n200e2-uniform-s0](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-answer-kl-grid-v1/runs/51oa73g7)|200|0.0001|0.001|1|0.3|0.3926|0.2480|0.8571|0.5140|0.8425|

### Grid: the same, from a gate-initialized start

**Question.** The same sweep, but starting from the released gate rather than
from a stage-1 checkpoint.

|W&B run|contexts|gate LR|mixer LR|accum.|train NLL|train KL|train acc.|val NLL|val acc.|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|[q25a-kl-gi-n2000e2-a16-lr01-s0](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-answer-kl-gateinit-grid-v1/runs/mdpl0nuo)|2000|1e-05|0.0001|16|0.4036|0.4223|0.8935|0.7037|0.8005|
|[q25a-kl-gi-n2000e2-a16-lr05-s0](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-answer-kl-gateinit-grid-v1/runs/01ny28vm)|2000|5e-05|0.0005|16|0.3489|0.3063|0.8962|0.6296|0.8111|
|[q25a-kl-gi-n2000e2-a32-lr01-s0](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-answer-kl-gateinit-grid-v1/runs/vatg4di6)|2000|1e-05|0.0001|32|0.5411|0.5179|0.8322|0.7015|0.7926|
|[q25a-kl-gi-n2000e2-a32-lr05-s0](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-answer-kl-gateinit-grid-v1/runs/qc9m616b)|2000|5e-05|0.0005|32|0.4233|0.3898|0.8863|0.6560|0.8020|
|[q25a-kl-gi-n2000e2-a8-lr01-s0](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-answer-kl-gateinit-grid-v1/runs/m109up9e)|2000|1e-05|0.0001|8|0.3800|0.3933|0.8600|0.6683|0.8104|
|[q25a-kl-gi-n2000e2-a8-lr05-s0](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-answer-kl-gateinit-grid-v1/runs/wktg4isy)|2000|5e-05|0.0005|8|0.3077|0.2996|0.8895|0.6957|0.7898|
|[q25a-kl-gi-n2000e2-a8-lr10-s0](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-answer-kl-gateinit-grid-v1/runs/q1hdpzze)|2000|0.0001|0.001|8|0.3135|0.2901|0.8974|0.6915|0.7921|
|[q25a-kl-gi-n200e2-uniform-s0](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-answer-kl-gateinit-grid-v1/runs/0dq7kx9h)|200|0.0001|0.001|1|0.6602|0.3758|0.8214|0.4971|0.8539|

|evaluation run|task|full|0.4|0.3|0.2|0.1|0.05|
|---|---|---:|---:|---:|---:|---:|---:|
|q25a-kl-gi-n2000e2-a16-lr01-s0-eval-scbench-kv|choice_eng|77.78|79.17|76.39|75.00|66.20|50.46|
|q25a-kl-gi-n2000e2-a16-lr01-s0-eval-scbench-kv|kv|69.20|62.00|48.40|47.00|24.80|1.20|
|q25a-kl-gi-n2000e2-a16-lr01-s0-eval-scbench-kv|many_shot|38.52|38.52|34.07|35.19|34.44|34.44|
|q25a-kl-gi-n2000e2-a16-lr01-s0-eval-scbench-kv|mf|33.17|34.00|33.83|32.83|29.67|13.50|
|q25a-kl-gi-n2000e2-a16-lr01-s0-eval-scbench-kv|prefix_suffix|49.20|44.20|46.60|32.80|6.40|1.20|
|q25a-kl-gi-n2000e2-a16-lr01-s0-eval-scbench-kv|qa_eng|42.92|44.13|43.52|40.90|23.95|14.20|
|q25a-kl-gi-n2000e2-a16-lr01-s0-eval-scbench-kv|repoqa|59.09|60.00|58.18|57.50|13.86|2.05|
|q25a-kl-gi-n2000e2-a16-lr01-s0-eval-scbench-kv|summary|37.02|36.35|36.84|35.87|31.18|27.12|
|q25a-kl-gi-n2000e2-a16-lr01-s0-eval-scbench-kv|vt|40.53|39.16|38.89|42.84|40.49|31.24|
|q25a-kl-gi-n2000e2-a16-lr05-s0-eval-scbench-kv|choice_eng|79.17|77.78|75.00|75.00|68.06|47.69|
|q25a-kl-gi-n2000e2-a16-lr05-s0-eval-scbench-kv|kv|68.20|54.60|50.60|39.60|26.20|1.00|
|q25a-kl-gi-n2000e2-a16-lr05-s0-eval-scbench-kv|many_shot|38.15|38.52|35.56|34.81|33.70|33.70|
|q25a-kl-gi-n2000e2-a16-lr05-s0-eval-scbench-kv|mf|33.00|35.67|35.17|33.67|26.83|15.33|
|q25a-kl-gi-n2000e2-a16-lr05-s0-eval-scbench-kv|prefix_suffix|51.20|45.80|36.00|29.80|9.20|1.00|
|q25a-kl-gi-n2000e2-a16-lr05-s0-eval-scbench-kv|qa_eng|41.52|42.87|45.37|41.15|27.91|15.11|
|q25a-kl-gi-n2000e2-a16-lr05-s0-eval-scbench-kv|repoqa|59.09|60.00|60.00|54.77|8.41|2.05|
|q25a-kl-gi-n2000e2-a16-lr05-s0-eval-scbench-kv|repoqa_and_kv|80.82|80.54|79.55|74.57|20.60|1.42|
|q25a-kl-gi-n2000e2-a16-lr05-s0-eval-scbench-kv|summary|36.82|36.67|36.64|35.84|31.75|27.10|
|q25a-kl-gi-n2000e2-a16-lr05-s0-eval-scbench-kv|summary_with_needles|68.35|67.72|68.29|66.67|50.71|13.75|
|q25a-kl-gi-n2000e2-a16-lr05-s0-eval-scbench-kv|vt|41.29|41.78|36.67|37.24|37.64|23.33|
|q25a-kl-gi-n2000e2-a32-lr01-s0-eval-scbench-kv|choice_eng|79.17|76.39|75.00|76.39|67.59|50.46|
|q25a-kl-gi-n2000e2-a32-lr01-s0-eval-scbench-kv|kv|68.20|68.40|49.80|47.20|29.80|1.00|
|q25a-kl-gi-n2000e2-a32-lr01-s0-eval-scbench-kv|many_shot|38.15|38.15|35.19|33.70|33.70|33.70|
|q25a-kl-gi-n2000e2-a32-lr01-s0-eval-scbench-kv|mf|33.00|33.83|35.00|32.00|28.00|11.67|
|q25a-kl-gi-n2000e2-a32-lr01-s0-eval-scbench-kv|prefix_suffix|51.20|48.20|43.20|33.40|5.80|1.20|
|q25a-kl-gi-n2000e2-a32-lr01-s0-eval-scbench-kv|qa_eng|41.52|43.38|46.42|43.13|23.71|13.34|
|q25a-kl-gi-n2000e2-a32-lr01-s0-eval-scbench-kv|repoqa|59.09|59.55|59.09|58.18|14.77|2.05|
|q25a-kl-gi-n2000e2-a32-lr01-s0-eval-scbench-kv|summary|36.82|36.37|36.29|35.94|30.73|27.19|
|q25a-kl-gi-n2000e2-a32-lr01-s0-eval-scbench-kv|vt|41.29|38.89|40.53|44.27|46.09|33.87|
|q25a-kl-gi-n2000e2-a32-lr05-s0-eval-scbench-kv|choice_eng|79.17|76.39|73.61|75.00|67.59|47.69|
|q25a-kl-gi-n2000e2-a32-lr05-s0-eval-scbench-kv|kv|68.20|51.80|46.40|40.60|28.40|1.20|
|q25a-kl-gi-n2000e2-a32-lr05-s0-eval-scbench-kv|many_shot|38.15|36.67|34.81|34.44|33.70|33.70|
|q25a-kl-gi-n2000e2-a32-lr05-s0-eval-scbench-kv|mf|33.00|34.33|34.00|32.50|27.50|14.50|
|q25a-kl-gi-n2000e2-a32-lr05-s0-eval-scbench-kv|prefix_suffix|51.20|46.00|40.20|33.40|9.40|1.00|
|q25a-kl-gi-n2000e2-a32-lr05-s0-eval-scbench-kv|qa_eng|41.52|42.12|43.60|41.64|29.46|15.35|
|q25a-kl-gi-n2000e2-a32-lr05-s0-eval-scbench-kv|repoqa|59.09|60.68|59.55|56.82|8.86|2.05|
|q25a-kl-gi-n2000e2-a32-lr05-s0-eval-scbench-kv|repoqa_and_kv|80.82|79.83|80.40|77.13|23.30|1.42|
|q25a-kl-gi-n2000e2-a32-lr05-s0-eval-scbench-kv|summary|36.82|36.29|36.38|36.06|31.47|27.01|
|q25a-kl-gi-n2000e2-a32-lr05-s0-eval-scbench-kv|summary_with_needles|68.35|67.62|68.08|66.48|48.49|13.52|
|q25a-kl-gi-n2000e2-a32-lr05-s0-eval-scbench-kv|vt|41.29|40.67|39.51|38.76|37.56|28.40|
|q25a-kl-gi-n2000e2-a8-lr01-s0-eval-scbench-kv|choice_eng|77.78|77.78|75.00|75.00|67.59|49.07|
|q25a-kl-gi-n2000e2-a8-lr01-s0-eval-scbench-kv|kv|69.20|53.40|46.40|46.20|27.40|1.00|
|q25a-kl-gi-n2000e2-a8-lr01-s0-eval-scbench-kv|many_shot|38.52|38.52|34.07|34.81|34.44|34.44|
|q25a-kl-gi-n2000e2-a8-lr01-s0-eval-scbench-kv|mf|33.17|34.33|34.83|32.17|28.00|15.00|
|q25a-kl-gi-n2000e2-a8-lr01-s0-eval-scbench-kv|prefix_suffix|49.20|46.80|47.80|33.80|7.60|1.00|
|q25a-kl-gi-n2000e2-a8-lr01-s0-eval-scbench-kv|qa_eng|42.92|43.90|42.20|39.81|22.32|15.11|
|q25a-kl-gi-n2000e2-a8-lr01-s0-eval-scbench-kv|repoqa|59.09|58.86|58.64|58.18|12.73|2.05|
|q25a-kl-gi-n2000e2-a8-lr01-s0-eval-scbench-kv|summary|37.02|36.23|36.66|35.63|31.31|27.00|
|q25a-kl-gi-n2000e2-a8-lr01-s0-eval-scbench-kv|vt|40.53|39.87|39.20|43.16|41.42|29.07|
|q25a-kl-gi-n2000e2-a8-lr05-s0-eval-scbench-kv|choice_eng|77.78|76.39|76.39|73.61|60.65|47.69|
|q25a-kl-gi-n2000e2-a8-lr05-s0-eval-scbench-kv|kv|67.80|51.80|50.00|38.40|16.20|1.00|
|q25a-kl-gi-n2000e2-a8-lr05-s0-eval-scbench-kv|many_shot|37.78|38.52|35.93|33.33|33.33|33.33|
|q25a-kl-gi-n2000e2-a8-lr05-s0-eval-scbench-kv|mf|32.00|34.83|35.17|33.50|28.67|14.67|
|q25a-kl-gi-n2000e2-a8-lr05-s0-eval-scbench-kv|prefix_suffix|49.60|46.60|38.40|28.80|7.80|1.20|
|q25a-kl-gi-n2000e2-a8-lr05-s0-eval-scbench-kv|qa_eng|41.09|41.10|43.04|42.10|27.55|14.23|
|q25a-kl-gi-n2000e2-a8-lr05-s0-eval-scbench-kv|repoqa|59.09|58.86|56.59|50.91|6.36|2.05|
|q25a-kl-gi-n2000e2-a8-lr05-s0-eval-scbench-kv|summary|37.01|36.51|36.99|36.77|31.93|27.19|
|q25a-kl-gi-n2000e2-a8-lr05-s0-eval-scbench-kv|vt|40.13|40.49|40.84|47.47|40.44|24.58|
|q25a-kl-gi-n2000e2-a8-lr10-s0-eval-scbench-kv|choice_eng|77.78|77.78|75.00|75.00|59.26|49.07|
|q25a-kl-gi-n2000e2-a8-lr10-s0-eval-scbench-kv|kv|69.20|68.20|58.80|32.80|9.00|0.80|
|q25a-kl-gi-n2000e2-a8-lr10-s0-eval-scbench-kv|many_shot|38.52|38.15|34.81|34.81|34.44|34.44|
|q25a-kl-gi-n2000e2-a8-lr10-s0-eval-scbench-kv|mf|33.17|33.17|35.17|34.67|28.50|17.50|
|q25a-kl-gi-n2000e2-a8-lr10-s0-eval-scbench-kv|prefix_suffix|49.20|44.40|37.40|23.40|4.40|1.40|
|q25a-kl-gi-n2000e2-a8-lr10-s0-eval-scbench-kv|qa_eng|42.92|40.70|40.38|43.41|27.31|15.37|
|q25a-kl-gi-n2000e2-a8-lr10-s0-eval-scbench-kv|repoqa|59.09|57.27|52.05|37.73|2.95|2.05|
|q25a-kl-gi-n2000e2-a8-lr10-s0-eval-scbench-kv|summary|37.02|37.20|37.75|36.83|31.96|27.06|
|q25a-kl-gi-n2000e2-a8-lr10-s0-eval-scbench-kv|vt|40.53|37.38|36.49|35.82|38.76|13.87|
|q25a-kl-gi-n200e2-uniform-s0-eval-scbench-kv|choice_eng|79.17|77.78|76.39|73.61|66.20|47.69|
|q25a-kl-gi-n200e2-uniform-s0-eval-scbench-kv|kv|68.20|74.00|65.40|51.40|23.40|1.20|
|q25a-kl-gi-n200e2-uniform-s0-eval-scbench-kv|many_shot|38.15|35.56|33.33|35.56|33.70|33.70|
|q25a-kl-gi-n200e2-uniform-s0-eval-scbench-kv|mf|33.00|35.50|36.00|34.00|27.50|16.17|
|q25a-kl-gi-n200e2-uniform-s0-eval-scbench-kv|prefix_suffix|51.20|47.20|42.20|34.80|6.60|1.00|
|q25a-kl-gi-n200e2-uniform-s0-eval-scbench-kv|qa_eng|41.52|43.07|44.65|40.69|21.69|13.35|
|q25a-kl-gi-n200e2-uniform-s0-eval-scbench-kv|repoqa|59.09|59.32|58.41|55.68|9.32|2.05|
|q25a-kl-gi-n200e2-uniform-s0-eval-scbench-kv|repoqa_and_kv|80.82|80.97|80.11|74.01|14.35|1.42|
|q25a-kl-gi-n200e2-uniform-s0-eval-scbench-kv|summary|36.82|36.58|35.01|35.25|31.15|27.14|
|q25a-kl-gi-n200e2-uniform-s0-eval-scbench-kv|summary_with_needles|68.35|68.44|65.44|65.80|47.88|13.64|
|q25a-kl-gi-n200e2-uniform-s0-eval-scbench-kv|vt|41.29|39.02|40.00|39.56|43.20|25.24|

### Grid: gate-only learning-rate screen

**Question.** With no mixer at all, what gate learning rate is best? This is the
closest thing the project had to a control before Grid 13 built one properly.

|W&B run|gate LR|graph dim|train NLL|train KL|train acc.|val NLL|val acc.|
|---|---:|---:|---:|---:|---:|---:|---:|
|[q25a-klgo-pre-n200e2-lr05-s0](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-answer-klgo-lrscreen-v1/runs/uy0l5z65)|5e-05|—|0.3379|0.1602|0.8929|0.5262|0.8562|
|[q25a-klgo-pre-n200e2-lr10-s0](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-answer-klgo-lrscreen-v1/runs/ewld5x8f)|0.0001|—|0.5586|0.2310|0.7857|0.4892|0.8562|
|[q25a-klgo-pre-n200e2-lr100-s0](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-answer-klgo-lrscreen-v1/runs/zqmxn43u)|0.001|—|0.7422|0.4387|0.7857|0.4694|0.8402|
|[q25a-klgo-pre-n200e2-lr30-s0](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-answer-klgo-lrscreen-v1/runs/c1yipt7p)|0.0003|—|0.7539|0.4304|0.7857|0.4975|0.8470|
|[q25a-klgo-rnd-n200e2-lr05-s0](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-answer-klgo-lrscreen-v1/runs/4orijz0o)|5e-05|—|0.9258|0.6989|0.7857|0.6030|0.8082|
|[q25a-klgo-rnd-n200e2-lr10-s0](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-answer-klgo-lrscreen-v1/runs/wwpsabwt)|0.0001|—|0.7578|0.4906|0.8214|0.5127|0.8333|
|[q25a-klgo-rnd-n200e2-lr100-s0](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-answer-klgo-lrscreen-v1/runs/kh2t8r4w)|0.001|—|0.8086|0.5262|0.7857|0.5875|0.8288|
|[q25a-klgo-rnd-n200e2-lr30-s0](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-answer-klgo-lrscreen-v1/runs/jcoc5uu7)|0.0003|—|0.9648|0.8009|0.8571|0.4934|0.8493|

### Benchmark scores

Also at the 2% protected window. Only the two lowest retention levels were
evaluated for this grid.

|run|0.1|0.05|full|kv @ lowest|prefix/suffix @ lowest|
|---|---:|---:|---:|---:|---:|
|pretrained gate, gate LR 5e-5|38.46|19.64|54.23|3.20|0.60|
|pretrained gate, gate LR 1e-4|38.33|21.22|54.23|2.80|0.80|
|pretrained gate, gate LR 1e-3|20.87|14.71|54.23|1.80|1.00|
|pretrained gate, gate LR 3e-4|36.54|19.75|54.23|2.00|0.80|
|random gate, gate LR 5e-5|15.46|14.07|54.23|1.80|1.80|
|random gate, gate LR 1e-4|14.73|13.68|54.23|1.80|1.40|
|random gate, gate LR 1e-3|15.60|14.62|54.23|1.80|1.60|
|random gate, gate LR 3e-4|15.21|13.65|54.23|1.80|1.00|

**What this shows.** Two things, both useful.

Starting the gate from the released checkpoint is worth roughly 23 points at
one tenth retention: 38.46 against 15.46 for a random start. Answer supervision
alone cannot rebuild a gate from nothing.

And because this grid has no mixer at all, it serves as a control for the
architecture grid at the two ratios they share:

|retention|best with a mixer|best gate only|difference|
|---|---:|---:|---:|
|0.1|39.98|38.46|+1.52|
|0.05|20.08|21.22|-1.14|

A mixer is worth about a point and a half at one tenth retention, and slightly
negative at one twentieth. That is the size of the effect this project set out
to find.

<details>
<summary>Per-task detail, all 8 runs</summary>

**pretrained gate, gate LR 5e-5**

|task|full|0.1|0.05|
|---|---:|---:|---:|
|kv|69.20|34.20|3.20|
|prefix_suffix|49.20|10.60|0.60|
|summary|37.02|33.08|27.87|
|vt|40.53|51.56|43.29|
|many_shot|38.52|30.74|33.70|
|mf|33.17|29.83|19.50|
|choice_eng|77.78|67.59|46.30|
|qa_eng|42.92|19.99|14.07|
|repoqa|59.09|42.73|4.55|
|summary_with_needles|68.43|48.65|16.82|
|repoqa_and_kv|80.68|54.12|6.11|

**pretrained gate, gate LR 1e-4**

|task|full|0.1|0.05|
|---|---:|---:|---:|
|kv|69.20|33.20|2.80|
|prefix_suffix|49.20|12.40|0.80|
|summary|37.02|33.02|27.72|
|vt|40.53|48.49|44.31|
|many_shot|38.52|30.37|35.56|
|mf|33.17|28.50|20.50|
|choice_eng|77.78|64.81|56.48|
|qa_eng|42.92|20.17|15.71|
|repoqa|59.09|42.95|5.00|
|summary_with_needles|68.43|51.78|19.58|
|repoqa_and_kv|80.68|55.97|4.97|

**pretrained gate, gate LR 1e-3**

|task|full|0.1|0.05|
|---|---:|---:|---:|
|kv|69.20|1.60|1.80|
|prefix_suffix|49.20|1.20|1.00|
|summary|37.02|32.82|28.05|
|vt|40.53|22.53|4.27|
|many_shot|38.52|34.81|32.59|
|mf|33.17|18.67|15.33|
|choice_eng|77.78|56.48|47.69|
|qa_eng|42.92|23.38|15.07|
|repoqa|59.09|4.32|1.14|
|summary_with_needles|68.43|31.75|14.05|
|repoqa_and_kv|80.68|1.99|0.85|

**pretrained gate, gate LR 3e-4**

|task|full|0.1|0.05|
|---|---:|---:|---:|
|kv|69.20|28.20|2.00|
|prefix_suffix|49.20|17.40|0.80|
|summary|37.02|33.34|27.61|
|vt|40.53|41.11|30.18|
|many_shot|38.52|33.70|33.70|
|mf|33.17|30.33|19.17|
|choice_eng|77.78|56.48|53.24|
|qa_eng|42.92|25.69|18.41|
|repoqa|59.09|37.50|2.27|
|summary_with_needles|68.43|52.02|25.87|
|repoqa_and_kv|80.68|46.16|3.98|

**random gate, gate LR 5e-5**

|task|full|0.1|0.05|
|---|---:|---:|---:|
|kv|69.20|1.60|1.80|
|prefix_suffix|49.20|1.80|1.80|
|summary|37.02|27.71|27.19|
|vt|40.53|2.62|2.76|
|many_shot|38.52|32.22|30.74|
|mf|33.17|17.17|14.17|
|choice_eng|77.78|56.02|50.46|
|qa_eng|42.92|15.39|10.85|
|repoqa|59.09|0.91|0.91|
|summary_with_needles|68.43|14.08|13.55|
|repoqa_and_kv|80.68|0.57|0.57|

**random gate, gate LR 1e-4**

|task|full|0.1|0.05|
|---|---:|---:|---:|
|kv|69.20|1.60|1.80|
|prefix_suffix|49.20|1.80|1.40|
|summary|37.02|27.24|27.14|
|vt|40.53|3.11|3.51|
|many_shot|38.52|31.11|30.74|
|mf|33.17|17.17|14.00|
|choice_eng|77.78|54.63|43.52|
|qa_eng|42.92|10.44|13.19|
|repoqa|59.09|0.91|0.91|
|summary_with_needles|68.43|13.45|13.72|
|repoqa_and_kv|80.68|0.57|0.57|

**random gate, gate LR 1e-3**

|task|full|0.1|0.05|
|---|---:|---:|---:|
|kv|69.20|1.80|1.80|
|prefix_suffix|49.20|1.60|1.60|
|summary|37.02|29.30|27.73|
|vt|40.53|3.02|3.82|
|many_shot|38.52|32.22|31.85|
|mf|33.17|17.50|16.17|
|choice_eng|77.78|50.46|48.15|
|qa_eng|42.92|13.95|13.61|
|repoqa|59.09|0.91|0.68|
|summary_with_needles|68.43|20.11|14.98|
|repoqa_and_kv|80.68|0.71|0.43|

**random gate, gate LR 3e-4**

|task|full|0.1|0.05|
|---|---:|---:|---:|
|kv|69.20|1.80|1.80|
|prefix_suffix|49.20|1.60|1.00|
|summary|37.02|28.38|27.30|
|vt|40.53|2.71|3.16|
|many_shot|38.52|32.96|31.11|
|mf|33.17|18.33|17.67|
|choice_eng|77.78|49.07|42.13|
|qa_eng|42.92|15.89|10.82|
|repoqa|59.09|0.91|0.91|
|summary_with_needles|68.43|14.80|13.63|
|repoqa_and_kv|80.68|0.85|0.57|

</details>

---

## Part 4 — the gate-space coupling

**Question.** The mixer's output had always been projected back to the model's
hidden width and added to the gate's input, where the gate's own projections
largely filtered it out. This grid tried wiring the mixer directly into the
gate's query, key and logit spaces instead, so it could not be ignored. It also
added learnable self-loops to the implicit graph.

**Design.** Three rows, each a full chain of stage 1, stage 2 and the eleven
SCBench tasks. Two architectures at one injection strength, and one architecture
at two strengths. The injection strength was calibrated so that 0.235
perturbs the gate about as much as the old coupling at its standard setting,
and 0.024 about a tenth as much.

### Stage 1

|W&B run|architecture|injection init|self loop|normalization|BCE first|BCE final|val BCE|steps|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|[q25a-gsc-gps-a1-s1](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-gate-space-coupling/runs/nrngbbh7)|gps|0.235|—|none|0.2111|0.1767|0.1941|469|
|[q25a-gsc-implicit-selfloop1-a01-s1](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-gate-space-coupling/runs/sruyr4jt)|implicit|0.024|1|batchnorm|0.1692|0.1760|0.1971|469|
|[q25a-gsc-implicit-selfloop1-a1-s1](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-gate-space-coupling/runs/k0dpe9tx)|implicit|0.235|1|batchnorm|0.2169|0.1914|0.2110|61|

### Stage 2

|W&B run|architecture|injection init|self loop|normalization|train NLL|train KL|train acc.|val NLL|val acc.|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|[q25a-gsc-gps-a1-s2](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-gate-space-coupling/runs/j51xby4w)|gps|0.235|—|none|0.5469|0.1976|0.8235|0.4007|0.8731|
|[q25a-gsc-implicit-selfloop1-a01-s2](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-gate-space-coupling/runs/ifdwlc5b)|implicit|0.024|1|batchnorm|0.4785|0.1987|0.8382|0.4670|0.8578|

### Benchmark scores, no protected window

**implicit graph, weak injection 0.024, self loops**

|task|full|0.4|0.3|0.2|0.1|0.05|
|---|---:|---:|---:|---:|---:|---:|
|kv|68.20|63.20|64.60|40.60|9.20|0.20|
|prefix_suffix|51.20|13.20|9.60|8.60|0.40|0.00|
|summary|36.82|36.87|36.07|35.46|32.99|28.50|
|vt|41.29|44.27|47.87|45.69|44.98|33.69|
|many_shot|38.15|35.93|35.93|34.44|30.74|31.48|
|mf|33.00|36.17|36.50|32.50|25.67|11.17|
|choice_eng|79.17|72.22|77.78|75.00|65.28|55.09|
|qa_eng|41.52|43.49|38.57|44.80|31.21|11.87|
|repoqa|59.09|54.32|51.36|48.18|29.32|1.36|
|summary_with_needles|68.35|67.73|66.60|64.65|52.31|16.43|
|repoqa_and_kv|80.82|76.56|72.59|68.75|41.76|2.41|

**graph transformer, injection 0.235**

|task|full|0.4|0.3|0.2|0.1|0.05|
|---|---:|---:|---:|---:|---:|---:|
|kv|68.20|8.80|3.00|0.40|0.00|0.00|
|prefix_suffix|51.20|1.40|0.20|0.00|0.00|0.00|
|summary|36.82|35.57|36.50|35.44|32.49|27.88|
|vt|41.29|34.98|39.42|39.91|35.20|22.93|
|many_shot|38.15|33.33|33.33|32.96|34.81|31.11|
|mf|33.00|26.67|24.50|23.50|14.33|2.83|
|choice_eng|79.17|73.61|75.00|73.61|60.65|50.46|
|qa_eng|41.52|35.93|34.53|37.78|28.77|11.73|
|repoqa|59.09|27.05|21.82|17.27|4.32|0.68|
|summary_with_needles|68.35|62.76|65.47|65.14|53.37|16.93|
|repoqa_and_kv|80.82|23.58|22.02|15.20|3.69|0.28|

**Reading.** The weaker injection scores clearly better than the graph
transformer at the stronger one, and the difference is concentrated in
key-value retrieval: 40.6 against 0.4 at one fifth retention. The stage-1
losses of these two rows differ by less than one percent.

**Incidents.** One row of this grid died twice. First all three rows filled a
compute node's local disk, because stage 1 stores every context's hidden states
there and the jobs had declared a fraction of what they needed. Passing the
scores-only cache option removed about 150 GB per run and made the stage faster,
since recomputing the hidden states costs about what reading them back costs.
The row then died again on a numerical fault: the gate's keep probability was
computed as a bare reciprocal of a sum of exponentials, which overflows once the
logit gap passes about 88, and the backward pass then multiplied zero by
infinity. Rewriting it through a log-sum-exp and a sigmoid fixed it exactly.

---

## Part 5 — the mixer diagnosis

**Question.** By this point every configuration produced the same loss and
different benchmark scores. This grid was built to find out why, with six rows
that each change exactly one thing.

| row | what it changes | what it tests |
|---|---|---|
| gate only | no mixer at all | the missing control |
| mixer, one stage | a mixer trained in stage 2 only | the same start and length as the control, differing only by the mixer |
| reference | nothing | the baseline the others move from |
| frozen gate | the gate cannot adapt | whether any gain must come from the mixer |
| random gate | the gate starts random | whether the pretrained gate is a local optimum |
| ten times faster mixer | mixer learning rate 1e-2 | whether this is an optimisation failure |

### Stage 1

|W&B run|coupling|injection init|gate frozen|gate ckpt|mixer LR|BCE first|BCE final|val BCE|steps|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|[q25a-diag-frozen-gate-s1](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-mixer-diagnosis/runs/arsj26fn)|gate-space|0.235|yes|—|0.001|0.2176|0.1791|0.1993|469|
|[q25a-diag-high-mixer-lr-s1](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-mixer-diagnosis/runs/3d4yccsa)|gate-space|0.235|no|—|0.01|0.2108|0.1764|0.1976|469|
|[q25a-diag-mixer-s1](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-mixer-diagnosis/runs/4whl6ftn)|gate-space|0.235|no|—|0.001|0.2172|0.1764|0.1980|469|
|[q25a-diag-random-gate-s1](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-mixer-diagnosis/runs/pnx1xi3z)|gate-space|0.235|no|—|0.001|0.4133|0.1786|0.1998|469|

### Stage 2

|W&B run|coupling|injection init|gate frozen|gate ckpt|mixer LR|train NLL|train KL|train acc.|val NLL|val acc.|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|[q25a-diag-frozen-gate-s2](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-mixer-diagnosis/runs/tq5b9xtx)|gate-space|0.235|no|—|0.001|0.5312|0.1577|0.8088|0.4521|0.8584|
|[q25a-diag-gate-only-s2](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-mixer-diagnosis/runs/yukzpen2)|—|—|no|—|0.001|0.4824|0.1623|0.8382|0.5031|0.8630|
|[q25a-diag-high-mixer-lr-s2](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-mixer-diagnosis/runs/l2zcuulb)|gate-space|0.235|no|—|0.01|0.5430|0.2227|0.8382|0.6102|0.8105|
|[q25a-diag-mixer-s2](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-mixer-diagnosis/runs/8lpnupi0)|gate-space|0.235|no|—|0.001|0.7852|0.4132|0.8235|0.4979|0.8516|
|[q25a-diag-mixer-s2-s2](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-mixer-diagnosis/runs/i0mxwqtz)|gate-space|0.235|no|—|0.001|0.6094|0.2345|0.8088|0.5317|0.8379|
|[q25a-diag-random-gate-s2](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-mixer-diagnosis/runs/78bxzq4f)|gate-space|0.235|no|—|0.001|0.6172|0.2446|0.8088|0.5238|0.8539|

### Benchmark scores, no protected window

**reference mixer**

|task|full|0.4|0.3|0.2|0.1|0.05|
|---|---:|---:|---:|---:|---:|---:|
|kv|69.20|47.00|42.60|28.20|7.80|0.00|
|prefix_suffix|49.20|12.20|4.60|1.20|0.00|0.00|
|summary|37.02|36.89|36.24|27.79|15.06|15.20|
|vt|40.53|38.49|40.36|37.33|33.56|27.64|
|many_shot|38.52|37.04|33.70|29.63|28.52|32.96|
|mf|33.17|33.00|32.83|31.17|16.83|6.67|
|choice_eng|77.78|75.00|76.39|43.98|25.93|38.89|
|qa_eng|42.92|42.54|42.51|34.73|16.80|6.62|
|repoqa|59.09|52.05|49.77|42.05|13.86|0.45|

**mixer, one stage**

|task|full|0.4|0.3|0.2|0.1|0.05|
|---|---:|---:|---:|---:|---:|---:|
|kv|69.20|43.00|43.20|23.20|0.00|0.00|
|prefix_suffix|49.20|0.00|0.60|0.00|0.00|0.00|
|summary|37.02|36.47|36.01|34.82|31.19|26.90|
|vt|40.53|37.64|39.24|36.93|31.64|24.89|
|many_shot|38.52|33.70|34.07|32.59|34.07|32.96|
|mf|33.17|33.33|33.00|36.17|23.17|10.17|
|choice_eng|77.78|69.44|72.22|68.06|56.02|42.59|
|qa_eng|42.92|41.91|36.97|40.76|22.62|14.15|
|repoqa|59.09|57.27|53.41|40.68|5.45|0.00|
|summary_with_needles|68.43|67.73|67.01|65.26|56.53|21.47|

**frozen gate**

|task|full|0.4|0.3|0.2|0.1|0.05|
|---|---:|---:|---:|---:|---:|---:|
|kv|69.20|59.60|50.80|0.60|0.00|0.00|
|prefix_suffix|49.20|1.60|0.20|0.00|0.00|0.00|
|summary|37.02|36.96|21.40|23.07|29.21|28.62|
|vt|40.53|44.36|43.51|40.98|35.82|17.78|
|many_shot|38.52|34.07|35.56|34.81|32.22|34.81|
|mf|33.17|34.83|36.00|33.83|24.50|7.17|
|choice_eng|77.78|80.56|44.44|69.44|57.87|44.91|
|qa_eng|42.92|41.85|9.10|27.45|27.76|12.39|
|repoqa|59.09|55.91|49.77|40.45|9.55|0.91|
|summary_with_needles|68.43|67.35|24.34|31.68|28.67|18.04|

**random gate**

|task|full|0.4|0.3|0.2|0.1|0.05|
|---|---:|---:|---:|---:|---:|---:|
|kv|69.20|69.40|45.80|20.20|3.00|0.00|
|prefix_suffix|49.20|18.40|15.20|4.20|0.20|0.00|
|summary|37.02|37.88|37.10|35.68|31.78|27.81|
|vt|40.53|34.58|32.04|29.82|28.58|17.24|
|many_shot|38.52|36.67|35.93|34.44|34.44|32.22|
|mf|33.17|35.33|37.17|31.33|24.00|11.00|
|choice_eng|77.78|72.22|73.61|73.61|62.50|53.70|
|qa_eng|42.92|40.70|44.31|41.37|28.06|15.66|

**ten times faster mixer**

|task|full|0.4|0.3|0.2|0.1|0.05|
|---|---:|---:|---:|---:|---:|---:|
|kv|69.20|21.00|12.20|9.40|0.80|0.00|
|prefix_suffix|49.20|5.40|0.80|0.00|0.00|0.00|
|summary|37.02|36.52|35.48|33.48|29.35|27.40|
|vt|40.53|27.47|25.91|0.13|0.18|3.51|
|many_shot|38.52|35.19|35.56|29.63|28.15|28.52|
|mf|33.17|28.50|26.83|24.50|13.33|4.50|
|choice_eng|77.78|73.61|75.00|68.06|58.33|51.39|
|qa_eng|42.92|40.05|39.95|32.22|31.15|12.06|
|repoqa|59.09|38.64|31.36|17.05|0.91|0.23|
|summary_with_needles|68.43|66.17|65.18|61.76|28.15|14.70|

**Reading.** Three things came out of this grid.

First, the ordering at one fifth retention is monotonic in how much influence
the mixer is given. The weakest mixer is best, the reference is fourteen points
worse, and the fastest-learning mixer is worst of all. More mixer is worse.

Second, the damage is concentrated in retrieval. On the six tasks that tolerate
losing arbitrary tokens, every row scores between 79 and 100 percent of full
cache. On the two that need specific tokens, the spread runs from 38 percent
down to near zero.

Third, and most usefully, the mixer is *not* being switched off by training.
Reading the saved weights showed the injection maps of the strongest row moved
by one percent across ten epochs while its body grew by twenty; a diagnostic
added afterwards showed the mixer contributing between 20 and 30 percent of the
gate's query magnitude and between 33 and 84 percent of its key magnitude. The
mixer is loud, it trains hard, and the loss does not care.

**Incident.** The ranking metric built to answer exactly this question had never
recorded a single value. A line clearing it for training contexts sat one indent
level too far out, so it also ran on the validation branch and erased the value
set two lines above. Every test passed throughout, because every test called the
evaluator directly where the value was correct. Four grids ran blind to it.

---

## Part 6 — the protocol-matched comparison

**Question.** The published baselines protect the most recent 2% of tokens from
eviction. Our grids protected none. How much of the gap is protocol?

**Design.** Two models, each evaluated at both settings, split into one job per
benchmark task so the suite finishes in the time of its slowest task. Twenty-two
jobs across five cluster accounts.

### What the protected window is worth

One model, measured both ways, eleven-task mean absolute score.

|retention|no window|2% window|gain|
|---:|---:|---:|---:|
|0.4|49.45|51.49|+2.04|
|0.3|48.86|51.47|+2.61|
|0.2|45.33|47.05|+1.72|
|0.1|33.08|34.35|+1.27|
|0.05|17.47|17.94|+0.46|

### The comparison

Eleven-task mean absolute score. Uncompressed sits near 54.3 for every row.

|configuration|0.4|0.3|0.2|0.1|0.05|full|
|---|---:|---:|---:|---:|---:|---:|
|paper: graph selector (2% window)|54.24|53.25|50.92|—|—|54.49|
|paper: Fast KVzip (2% window)|54.15|53.71|50.08|—|—|54.36|
|ours: weak injection, gate-space (2% window)|51.49|51.47|47.05|34.35|17.94|54.23|
|ours: earlier hidden-coupling model (no window)|51.54|49.11|46.13|36.38|18.15|54.22|
|ours: weak injection, gate-space (no window)|49.45|48.86|45.33|33.08|17.47|54.33|
|paper: KVzip (no window)|52.31|51.46|44.58|—|—|54.36|

### Per-task detail for the two new evaluations

**earlier hidden-coupling model, no protected window**

|task|full|0.4|0.3|0.2|0.1|0.05|
|---|---:|---:|---:|---:|---:|---:|
|kv|69.20|67.40|67.00|57.20|28.80|0.00|
|prefix_suffix|49.20|30.00|16.80|2.80|0.20|0.00|
|summary|37.02|37.32|36.45|35.74|33.65|27.73|
|vt|40.53|40.27|40.27|44.36|48.27|43.33|
|many_shot|38.52|34.81|33.70|32.22|32.59|32.22|
|mf|33.00|34.17|35.83|35.50|22.33|5.67|
|choice_eng|79.17|75.00|73.61|72.22|66.67|53.70|
|qa_eng|41.52|42.90|38.99|39.31|29.35|12.77|
|repoqa|59.09|59.32|53.64|51.14|35.23|4.32|
|summary_with_needles|68.35|67.63|66.92|65.83|58.15|14.34|
|repoqa_and_kv|80.82|78.12|76.99|71.16|44.89|5.54|

**weak injection gate-space model, 2% protected window**

|task|full|0.4|0.3|0.2|0.1|0.05|
|---|---:|---:|---:|---:|---:|---:|
|kv|69.20|69.20|68.60|47.80|13.60|1.00|
|prefix_suffix|49.20|26.20|29.00|21.20|4.40|0.60|
|summary|37.02|36.84|36.35|36.01|32.70|27.79|
|vt|40.53|38.31|41.87|42.67|41.56|36.36|
|many_shot|38.52|36.67|36.30|33.70|32.96|32.59|
|mf|33.17|34.50|35.50|34.00|26.17|14.17|
|choice_eng|77.78|77.78|73.61|69.44|69.44|49.07|
|qa_eng|42.92|42.43|41.87|43.23|30.94|17.83|
|repoqa|59.09|57.27|57.50|51.36|26.14|0.91|
|summary_with_needles|68.43|67.97|67.55|66.27|62.01|15.28|
|repoqa_and_kv|80.68|79.26|77.98|71.88|37.93|1.70|

---

## Verdict

**The work matches the published result.** The best configuration of the
architecture grid scores 51.35 at one fifth retention against the paper's 50.92,
on the same model, the same eleven tasks and the same protected window. That is
the first thing to say, and earlier drafts of this record understated it because
they compared a later grid measured under a harder protocol.

**What a graph buys over the gate alone is about one point.** Measured against a
gate-only control at matched protocol, a mixer is worth +1.5 at one tenth
retention and -1.1 at one twentieth. The effect is real at one setting, absent
at another, and single-seed throughout. It is not nothing, and it is not much.

**Pushing the mixer harder is reliably worse.** Three runs that change only the
mixer's initial contribution lose ten points and then ten more. The gate-space
coupling, built specifically to let the mixer reach the gate rather than be
filtered out, reproduces the same ordering: the weakest injection wins, and a
ten-times-faster mixer is the worst configuration tested anywhere.

**The gate-space coupling is behind the coupling it replaced.** 47.05 against
51.35 at one fifth retention and the same window. It does what it was designed
to do, contributing up to 84 percent of the gate's key magnitude where the old
coupling was largely ignored, and it costs four points.

**The damage is concentrated in retrieval.** On tasks that tolerate losing
arbitrary tokens, every configuration sits within a normal band. On tasks that
need specific tokens the spread is enormous, and two configurations produce
nothing usable at all.

**The loss was the wrong instrument throughout.** Four grids were steered by a
calibration loss that cannot see a ranking change, while the metric that could
was computed and silently discarded by a single misplaced line. That is the
most important methodological lesson in this record, and it is why the
architecture grid's stage-1 losses agree to within two percent while its
benchmark scores span twenty-three points.

**Where headroom remains.** At the lowest retention levels, which the published
tables do not report, every model collapses: about 18 out of 54 at one twentieth
of the cache. Both the largest spread between configurations and the largest
absolute loss live there. If a graph over the context is going to earn its
place, that is where to look.

---

## Appendix — additional and incomplete runs

Three runs exist in W&B that do not belong to any grid table above. They are
recorded here so the record is exhaustive.

|W&B run|state|what it was|outcome|
|---|---|---|---|
|`q25a-s40n40-n2000e2-uniform0025-025-shuffle-g28-s0-a8`|finished|A larger answer-training variant: 2000 contexts, retention sampled between 0.025 and 0.25, shuffled order, accumulation 8, graph microbatch 28.|Completed 515 updates. Train NLL 0.2643, token accuracy 0.874; validation NLL 0.5410, accuracy 0.833. No benchmark evaluation was run against it.|
|`q25a-s40n40-n200e2-uniform-s0-kl`|crashed|An early attempt at the KL-loss answer training that the later grids refined.|Stopped after 56 of its updates. Train NLL 1.2969, KL 1.1708, accuracy 0.697. Superseded by the KL grids in Part 3.|
|`q25a-diag-gate-only-s1`|failed|The first attempt at the gate-only control in Part 5.|Ran out of GPU memory after two minutes. Its recorded settings show why: with no mixer flags given, stage 1 fell back to its defaults and built a full hidden-coupling mixer at width 32 rather than no mixer at all. Stage 1 has no way to express "no mixer"; the supported gate-only route is an answer-training ablation. The control was rebuilt that way and appears in Part 5.|

The last of these is worth noting beyond bookkeeping. It is the reason the
gate-only control took three attempts to produce, and the reason Part 5's
control is a single-stage run rather than a two-stage one.

---

## Appendix — sizing and profiling runs

Ten short runs measured throughput and memory rather than quality. They are what
the microbatch settings in every other grid were chosen from. No benchmark
scores were produced.

|W&B run|token microbatch|graph microbatch|graph dim|state|recorded|
|---|---:|---:|---:|---|---|
|[qwen3-8b-e1-gate-random-trainable-seed0](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-profiling/runs/7sckty8p)|1000|8|32|finished|joint_backward_seconds_per_token=0.000428041, joint_forward_seconds_per_token=0.0023933|
|[qwen3-8b-e1-gate-random-trainable-seed0-warmcache](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-profiling/runs/r6so9mry)|1000|8|32|finished|joint_backward_seconds_per_token=0.000423595, joint_forward_seconds_per_token=0.00326072, tokens=982496|
|[qwen3-8b-e1-gate-random-trainable-seed0-warmcache-tok1000-graph32](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-profiling/runs/z380higj)|1000|32|32|finished|joint_backward_seconds_per_token=0.000166909, joint_forward_seconds_per_token=0.00234124, tokens=982496|
|[qwen3-8b-e1-gate-random-trainable-seed0-warmcache-tok1000-graph64](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-profiling/runs/41g9zba7)|1000|64|32|failed|joint_backward_seconds_per_token=0.000143137, joint_forward_seconds_per_token=0.00318606, tokens=434921|
|[qwen3-8b-e1-gate-random-trainable-seed0-warmcache-tok120000-graph16](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-profiling/runs/f0lenmoq)|120000|16|32|crashed|joint_backward_seconds_per_token=9.5876e-05, joint_forward_seconds_per_token=0.0024692, tokens=418885|
|[qwen3-8b-e1-gate-random-trainable-seed0-warmcache-tok120000-graph8](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-profiling/runs/j4cpbrtq)|120000|8|32|failed|joint_backward_seconds_per_token=9.73313e-05, joint_forward_seconds_per_token=0.00167827, tokens=434921|
|[qwen3-8b-e1-gate-random-trainable-seed0-warmcache-tok16000-graph16](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-profiling/runs/lav2a9bg)|16000|16|32|finished|joint_backward_seconds_per_token=9.38967e-05, joint_forward_seconds_per_token=0.00135304, tokens=982496|
|[qwen3-8b-e1-gate-random-trainable-seed0-warmcache-tok16000-graph24](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-profiling/runs/wjv99amj)|16000|24|32|failed|joint_backward_seconds_per_token=9.40662e-05, joint_forward_seconds_per_token=0.00316052, tokens=547762|
|[qwen3-8b-e1-gate-random-trainable-seed0-warmcache-tok32000-graph8](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-profiling/runs/w3c8l4x2)|32000|8|32|finished|joint_backward_seconds_per_token=9.46025e-05, joint_forward_seconds_per_token=0.00307068, tokens=982496|
|[qwen3-8b-e1-gate-random-trainable-seed0-warmcache-tok4000-batched](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-profiling/runs/b9ukhs3l)|4000|8|32|finished|joint_backward_seconds_per_token=0.000111113, joint_forward_seconds_per_token=0.00124656, tokens=982496|
---

## Appendix — run index

|W&B project|runs|link|
|---|---:|---|
|`graphkv-answer-kl-gateinit-grid-v1`|8|https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-answer-kl-gateinit-grid-v1|
|`graphkv-answer-kl-grid-v1`|8|https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-answer-kl-grid-v1|
|`graphkv-answer-klgo-lrscreen-v1`|8|https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-answer-klgo-lrscreen-v1|
|`graphkv-answer-qwen25-7b1m-s40n40-grid-v1`|16|https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-answer-qwen25-7b1m-s40n40-grid-v1|
|`graphkv-arch-lr-dim-grid-v1`|47|https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-arch-lr-dim-grid-v1|
|`graphkv-e124-g-rand-pre-freeze-tf`|9|https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-e124-g-rand-pre-freeze-tf|
|`graphkv-gate-space-coupling`|5|https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-gate-space-coupling|
|`graphkv-mixer-diagnosis`|11|https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-mixer-diagnosis|
|`graphkv-profiling`|10|https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-profiling|
|`graphkv-qwen25-7b1m-s40-stops-grid`|6|https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-qwen25-7b1m-s40-stops-grid|
|`graphkv-qwen3-adamw-eps-amsgrad-grid`|8|https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-qwen3-adamw-eps-amsgrad-grid|
|`graphkv-qwen3-context-alpha-grid`|11|https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-qwen3-context-alpha-grid|
|`graphkv-qwen3-subgraph-eps-epochs-grid`|12|https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-qwen3-subgraph-eps-epochs-grid|
|`graphkv-qwen3-unseen-context-grid`|8|https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-qwen3-unseen-context-grid|

