# Experiment configurations and results

This file is the experiment ledger for GraphKV.

The default tables describe the current command-line behavior. `Required` means
that the command must provide a value. `Not set` means that the feature is off.

## Default configurations

### Training: data and run control

| CLI option | Default value | Description |
|---|---:|---|
| `--model` | Required | Hugging Face model name or local model path. |
| `--output-dir` | `graph_checkpoints` | Root directory for checkpoints. |
| `--epochs` | `1` | Number of training epochs. |
| `--max-contexts` | Not set | Optional limit on training contexts per run. |
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
| `--alpha-init` | `0.1` | Initial learned mixer residual coefficient. |
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
| `--data` | `scbench_kv` | Dataset selector. For non-Instruct Qwen3, this resolves to `scbench_kv_short`. |
| `--idx` | `0` | First dataset example. |
| `--num` | `100` | Maximum number of examples. |
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
| `--wandb-project` | Not set | Training W&B project used for evaluation uploads. |
| `--wandb-entity` | Not set | Optional W&B entity. |

### Evaluation protocol used below

The tables below use the newer, comparable evaluation protocol.

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

## Results

### Grid 1: epochs, gate initialization, and gate freezing

This grid tested whether more epochs help, whether the released FastKVzip gate
is a better starting point, and whether that gate should remain frozen.
Random frozen gates are invalid, so the grid has nine runs.

All runs used one whole-context graph, joint training, `--alpha-init 0.1`, token microbatch `16000`, graph
microbatch `16`, no scheduler, no `best.pt`, and seed `0`.

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

This grid tests how AdamW epsilon interacts with training duration and the
number of independent subgraphs in each optimizer update. It crosses three
epsilon values, whole-context versus 2K-subgraph updates, and four versus eight
epochs.

All runs use 29 regular contexts, the trainable released gate, joint training,
`--alpha-init 0.1`, gate and mixer LR `1e-3`, token microbatch `16000`,
`--no-amsgrad`, and seed `0`. Both optimizers use 15% linear warmup followed by
cosine decay. No `best.pt` is saved. The three completed runs used graph
microbatch `8`; the nine resubmitted RTX Pro 6000 runs use graph microbatch
`16`.

`Running` means training is active. `Pending` means training is waiting for a
GPU. Scores replace these labels after evaluation finishes.

| W&B run | `--epochs` | `--adamw-eps` | `--subgraph-size` | `--subgraphs-per-step` | 1.00 (full) | 0.75 | 0.50 | 0.40 | 0.30 | 0.20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `train-qwen3-8b-e4-eps1e-4-fullctx-alpha01-seed0` | 4 | `1e-4` | Not set | Not used | Pending | Pending | Pending | Pending | Pending | Pending |
| [train-qwen3-8b-e4-eps1e-4-sg2k-step8-alpha01-seed0](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-qwen3-subgraph-eps-epochs-grid/runs/25wyodrx) | 4 | `1e-4` | 2000 | 8 | Running | Running | Running | Running | Running | Running |
| [train-qwen3-8b-e8-eps1e-4-fullctx-alpha01-seed0](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-qwen3-subgraph-eps-epochs-grid/runs/k7x9rsb3) | 8 | `1e-4` | Not set | Not used | Running | Running | Running | Running | Running | Running |
| [train-qwen3-8b-e8-eps1e-4-sg2k-step8-alpha01-seed0](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-qwen3-subgraph-eps-epochs-grid/runs/6t3wb8nm) | 8 | `1e-4` | 2000 | 8 | 66.70 | 66.00 | 66.10 | 65.90 | 62.80 | 51.40 |
| [train-qwen3-8b-e4-eps1e-6-fullctx-alpha01-seed0](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-qwen3-subgraph-eps-epochs-grid/runs/vzkbdf4s) | 4 | `1e-6` | Not set | Not used | Running | Running | Running | Running | Running | Running |
| `train-qwen3-8b-e4-eps1e-6-sg2k-step8-alpha01-seed0` | 4 | `1e-6` | 2000 | 8 | Pending | Pending | Pending | Pending | Pending | Pending |
| [train-qwen3-8b-e8-eps1e-6-fullctx-alpha01-seed0](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-qwen3-subgraph-eps-epochs-grid/runs/gen38wt6) | 8 | `1e-6` | Not set | Not used | 66.70 | 65.60 | 66.50 | 65.50 | 60.60 | 50.40 |
| [train-qwen3-8b-e8-eps1e-6-sg2k-step8-alpha01-seed0](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-qwen3-subgraph-eps-epochs-grid/runs/mvfmehyf) | 8 | `1e-6` | 2000 | 8 | 66.70 | 65.30 | 67.20 | 64.50 | 59.30 | 51.40 |
| [train-qwen3-8b-e4-eps1e-8-fullctx-alpha01-seed0](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-qwen3-subgraph-eps-epochs-grid/runs/a8ryhw15) | 4 | `1e-8` | Not set | Not used | Running | Running | Running | Running | Running | Running |
| `train-qwen3-8b-e4-eps1e-8-sg2k-step8-alpha01-seed0` | 4 | `1e-8` | 2000 | 8 | Pending | Pending | Pending | Pending | Pending | Pending |
| `train-qwen3-8b-e8-eps1e-8-fullctx-alpha01-seed0` | 8 | `1e-8` | Not set | Not used | Pending | Pending | Pending | Pending | Pending | Pending |
| [train-qwen3-8b-e8-eps1e-8-sg2k-step8-alpha01-seed0](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-qwen3-subgraph-eps-epochs-grid/runs/qegtkvaw) | 8 | `1e-8` | 2000 | 8 | Running | Running | Running | Running | Running | Running |
