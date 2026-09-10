# Experiment configurations and results

This file is the experiment ledger for GraphKV.

## Uniform-checkpoint full evaluation (2026-09-08, in progress)

The [approved RULER/full-evaluation plan](plans/ruler-evaluation.md) targets
`q25a-s40n40-n200e2-uniform-s0/best.pt`, window `0`, level `pair`, ratios
`0.75/0.50/0.40/0.30/0.20` and a full-cache baseline. Its 107 configurations
exclude Agentic. Submission order is SCBench KV, thirteen RULER 4K tasks,
thirteen RULER 8K tasks, then the remaining eighty configurations.

Local data smoke checks loaded one example from each of the thirteen pinned
4K splits, preserving task demonstrations, final questions/cues and reference
lists. These are loader checks, **not GPU pilot results or benchmark scores**.
Three-account staging and the corrected pilots are complete. All twelve pilot
attempts exited successfully; three original short-context executions were
invalidated and rerun after fixing explicit window zero. The
[pilot evidence](ruler-pilot-evidence.md) records all scores, outputs inspected,
timings and memory measurements. These are smoke tests, not benchmark results.
The user approved the [107-job production manifest](ruler-production-manifest.md)
and dashboard registration. **All 107 production jobs were submitted on
2026-09-08**, with the exact approved commands and submission order. Evaluation
is in progress; completed benchmark scores appear in the incremental snapshot below.

Read-only cluster checks authenticated `danieloh`, `guyzagor`, and `odedshah`.
The source checkpoint on `guyzagor` is 1,089,385,068 bytes and has SHA-256
`1f3e7c5354a966aa4409af9e609268e796fbec813e7b7d3c683843d9ba177ad0`.
Historical accounting confirms training job `21061828` and evaluation
`21061591` both completed with exit `0:0`; the latter took `04:11:34`.
The live scheduler returns `Invalid job id specified` for `21061828` on all
three identities. On 2026-09-08 the user explicitly approved omitting this expired
`afterok` dependency. The job remains checkpoint provenance; three-account
checkpoint verification and approval of the measured production manifest remain
required. Staging and pilots can proceed under this narrow exception.

The new evaluation-only W&B run is
`q25a-s40n40-n200e2-uniform-s0-eval-all-w0` in project
`graphkv-answer-qwen25-7b1m-s40n40-grid-v1`:
[hdo2z4x4](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-answer-qwen25-7b1m-s40n40-grid-v1/runs/hdo2z4x4).
It was created after pilots passed and initially contains configuration only.
Pilot jobs made no W&B writes; the source training run was not modified.
Old window-0.02 compressed scores are not exact regression targets for this
window-0 configuration.

### Retry redistribution (2026-09-08 20:59 UTC snapshot)

The [approved retry execution](ruler-retry-execution.md) submitted 51 replacements
across six accounts: Daniel 14, Guy 9, Oded 8, Dulberg 8, Shkabatu 7 and Liran 5.
Twenty-six failed originals resumed; 25 still-pending originals were cancelled
individually and replaced. Three originals had started and were left untouched.
The five partial evaluations remain in Daniel's original directories. No new
pilots were submitted; existing production measurements informed resource sizing.

All six checkpoint copies and isolated fixed runtimes were verified. Fifty
replacements use RTX 6000 with graph/token microbatches 8/16000; one uses the
reserved PRO queue with 16/16000. Checkpoint, evaluation protocol, W&B destination
and dashboard project are unchanged. The dashboard acknowledged all 51 new IDs.
The 20:59:30 UTC snapshot has 46 completed logical benchmarks, 24 running and
37 pending, with no replacement failures observed. These are scheduler counts,
not a claim that all newly completed results have already been uploaded. The
single retry-aware collector is collecting and uploading successful completions.

### Staging and first pilot attempts

All three isolated checkouts used clean commit `78d5d40` and identical checkpoint
bytes, prefix-token digest and runtime packages: PyTorch `2.7.0+cu128`,
Transformers `4.51.3`, Datasets `4.0.0`, FlashAttention `2.7.3`.
All resolved Qwen model caches use revision
`e28526f7bb80e2a9c8af03b831a9af3812f18fba`.

The actual saved architecture is **gate dimension 16 / sink keys 16**.
The `s40n40` tag refers to the source training-data start/count, not gate
dimensions; there is no architecture/name inconsistency. Structural checkpoint
validation and the source W&B configuration confirm these dimensions. The
saved model is Qwen2.5-7B-Instruct-1M, with 2K subgraphs, BF16, uniform retention
0.1–0.3, two epochs, 200 requested contexts and source W&B run `61ewyfcm`.
Weights and metadata were not changed.

| Account | Isolated checkout (checkpoint under `graph_checkpoints/answer/q25a-s40n40-n200e2-uniform-s0/best.pt`) | First pilot job IDs |
| --- | --- | --- |
| danieloh | `/home/danieloh/FastKVzip-ruler-evaluation` | `21112931`, `21112935`, `21112938` |
| guyzagor | `/home/guyzagor/danieloh/FastKVzip-ruler-evaluation` | `21112932`, `21112936`, `21112939` |
| odedshah | `/home/odedshah/FastKVzip-ruler-evaluation` | `21112933`, `21112937`, `21112942` |

All nine jobs completed with exit `0:0`, using one RTX Pro 6000 and 60 GiB host
RAM. They made no W&B writes. The six 128K RULER executions took 61–120 seconds
of task work per example, with 40.3–40.5 GiB peak allocated GPU memory and
29.1–30.8 GiB process peak RSS. These single-example measurements are sizing and
protocol evidence, not complete benchmark scores.

The first 4K pilots and three-context SQuAD timing check exposed a pre-existing
window helper behavior: it replaces integer zero with a protected 2% tail below
the prefill-chunk length. Thus jobs `21112931`, `21112932` and `21112942` do not
establish window-zero correctness. Corrected jobs `21113328`, `21113329` and
`21113330` completed with exit `0:0` at commit `7323c0d`, requesting 12G host
RAM and ten minutes based on initial measurements. Every corrected context
reports `Local window 0`; all retained-token diagnostics equal model-selected
retention. Original outputs/attempt IDs are retained. The six 128K executions
were unaffected. Old manifests remain readable but cannot resume into the new
window-revision-1 production protocol.

The complete inventory is 60,070 contexts and 138,483 questions: RULER has
39,000 contexts, SCBench 2,031, SQuAD 18,891, and filtered GSM 148 with the
verified tokenizer. SQuAD alone has 87,599 questions, so it is sized from its
own timing pilot rather than treated as a small dataset.

The complete grid is projected at 491.8 GPU-hours, with per-job time requests
totalling 915.25 hours and host memory ranging from 10G to 52G. These are modeled
estimates from small pilots and measured complete dataset sizes, not promises.
SQuAD alone is estimated at 56.1 hours (112.25-hour request). The proposed
five-slot-per-account load model assigns 36/37/34 jobs to
danieloh/guyzagor/odedshah. The exact order, resources, uncertainty and resolved
paths and all returned job IDs are in the production manifest. The first status
snapshot at 06:46 UTC reports **15 running (five per account), 92 pending**, with
pending reason `QOSMaxGRESPerUser`, not a dependency. SCBench KV `21114573`, all
thirteen RULER 4K tasks and the first RULER 8K task occupy those fifteen slots.
The remaining tiers were submitted after all earlier-tier IDs were captured,
without waiting for completion. No submission responses were ambiguous.

The serial collector polls exact IDs every five minutes, copies only successful
completed benchmark directories, and uploads through the existing completeness,
conflict and resumability checks to `hdo2z4x4`. It stops for investigation on a
job failure or connection error; it never automatically resubmits. Initial
coverage is **0/107 completed**, with no failures. Durable copies of the exact
manifest, approval/preflight records, all 107 attempt IDs and dashboard receipts
are retained in each account's grid directory.

All 405 prefill tests pass; compilation and shell syntax checks pass. The
bundled math/LaTeX suite has 356 passing and 429 failing tests, with the exact
same failing test IDs reproduced on clean `origin/main`; no math files changed.
The user explicitly approved dashboard job-ID export. The existing project
`graphkv-answer-qwen25-7b1m-s40n40-grid-v1` confirmed assignment of all **12 pilot
attempts plus 107 production IDs**. No checkpoint, model output or credential was
sent to that dashboard.

### Completed production results (incremental snapshot)

At **2026-09-08 08:27 UTC**, coverage is **25/107 benchmarks**; 15 jobs are
running and 67 remain queued, with no scheduler failures. The first complete benchmark, `ruler_qa_1_4k`
(job `21114589`), covers all 500 examples, the five compressed ratios and the
full-cache baseline. Its 17 score/relative/retention metric points were uploaded
to the new evaluation run `hdo2z4x4`; subsequent uploads brought the total to
425 points by 08:32 UTC without duplicating earlier uploads. No source training-run
metrics were changed. Every benchmark below covers 500/500 examples.
An independent raw-output audit recomputed all scores across 12,500 examples
and verified all 62,500 compressed retention records, with no discrepancies.

| Benchmark | Examples | Full | 0.75 | 0.50 | 0.40 | 0.30 | 0.20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ruler_niah_single_1_4k | 500/500 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| ruler_niah_single_2_4k | 500/500 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| ruler_niah_single_3_4k | 500/500 | 98.8 | 99.6 | 99.6 | 99.6 | 99.6 | 99.4 |
| ruler_niah_multikey_1_4k | 500/500 | 98.8 | 99.2 | 98.8 | 98.4 | 96.2 | 95.6 |
| ruler_niah_multikey_2_4k | 500/500 | 99.8 | 99.8 | 99.6 | 99.4 | 99.4 | 98.0 |
| ruler_niah_multikey_3_4k | 500/500 | 99.6 | 99.6 | 99.4 | 98.4 | 98.2 | 98.0 |
| ruler_niah_multivalue_4k | 500/500 | 91.90 | 92.85 | 85.45 | 81.15 | 75.30 | 62.85 |
| ruler_vt_4k | 500/500 | 98.88 | 98.56 | 96.00 | 95.40 | 93.68 | 90.08 |
| ruler_cwe_4k | 500/500 | 98.62 | 98.40 | 97.36 | 90.42 | 93.92 | 93.86 |
| ruler_fwe_4k | 500/500 | 85.80 | 84.60 | 82.60 | 83.67 | 85.40 | 84.07 |
| ruler_qa_1_4k | 500/500 | 84.8 | 85.2 | 85.8 | 85.6 | 86.0 | 81.6 |
| ruler_qa_2_4k | 500/500 | 59.6 | 59.4 | 59.4 | 61.2 | 61.0 | 60.8 |
| ruler_niah_single_1_8k | 500/500 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| ruler_niah_single_2_8k | 500/500 | 100.0 | 100.0 | 100.0 | 97.6 | 100.0 | 100.0 |
| ruler_niah_single_3_8k | 500/500 | 99.4 | 99.4 | 99.6 | 99.4 | 99.4 | 99.2 |
| ruler_niah_multikey_1_8k | 500/500 | 99.6 | 99.6 | 99.0 | 97.6 | 97.6 | 93.8 |
| ruler_niah_multikey_2_8k | 500/500 | 99.8 | 99.8 | 98.8 | 98.6 | 97.4 | 94.6 |
| ruler_niah_multikey_3_8k | 500/500 | 97.2 | 97.0 | 94.8 | 95.2 | 92.2 | 87.0 |
| ruler_niah_multiquery_8k | 500/500 | 99.90 | 99.95 | 99.90 | 99.85 | 99.90 | 99.60 |
| ruler_niah_multivalue_8k | 500/500 | 84.70 | 85.25 | 81.00 | 78.45 | 75.95 | 60.25 |
| ruler_vt_8k | 500/500 | 97.88 | 97.28 | 94.24 | 92.08 | 89.36 | 86.68 |
| ruler_cwe_8k | 500/500 | 91.02 | 90.68 | 90.10 | 34.92 | 43.08 | 85.70 |
| ruler_fwe_8k | 500/500 | 83.20 | 82.60 | 81.00 | 81.33 | 80.73 | 70.80 |
| ruler_qa_1_8k | 500/500 | 81.4 | 81.8 | 78.8 | 77.8 | 76.4 | 74.6 |
| ruler_qa_2_8k | 500/500 | 55.0 | 55.0 | 57.2 | 58.2 | 59.6 | 59.8 |

Mean actual retention equals model-selection retention at every ratio; means
for QA1 are 0.750000 / 0.499998 / 0.399990 / 0.299997 / 0.200000 in table order.
There is no protected-window contribution. QA1's scores show that this task
retains full-cache-level performance at 30–75%, with a 3.2-point drop at 20%.
Do not interpret small improvements over full cache as a general benefit of
compression before the other benchmarks finish. The two completed single-needle
tasks stay at 100 throughout; multikey-2 drops 1.8 points at 20% versus full
cache. QA2's compressed scores remain close to or slightly above its 59.6
full-cache score. At 4K, VT loses 8.8 points at 20%; FWE loses 1.73 points.
At 8K, these losses grow to 11.2 and 12.4 points respectively, while the completed
single-needle task remains perfect. Multivalue 4K has the largest drop so far:
29.05 points at 20%, despite near-full performance at 75%; this contrasts with
the near-perfect single-needle results. Coverage is **12/13 tasks at 4K and 13/13
at 8K**. Multiquery 4K is still running, so no 4K macro-average is claimed yet.

#### Complete RULER 8K macro-average

All thirteen tasks and 6,500 examples are complete. This is an equal-task mean
computed from unrounded scores and independently recomputed from all 39,000
full/compressed generated answers; it does not include any 4K or partial task.

| Length | Tasks | Full | 0.75 | 0.50 | 0.40 | 0.30 | 0.20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8K | 13/13 | 91.4692 | 91.4123 | 90.3415 | 85.4641 | 85.5095 | 85.5408 |

At 20% retention the macro score is 5.9285 points below full cache. At 50%, it
is only 1.1277 points below full cache. Actual mean context retention across all
6,500 examples is 0.750000 / 0.499999 / 0.399997 / 0.299999 / 0.199995, equal to
model selection at every compressed record, with no protected window.

Single-needle and multiquery retrieval remain nearly perfect. The largest
20%-versus-full losses are multivalue (24.45 points), FWE (12.40), VT (11.20),
and multikey-3 (10.20). QA2 improves by 4.80 points, while QA1 loses 6.80. These
are observed task outcomes, not evidence that compression generally improves
QA. The similar 20–40% macro scores conceal CWE's pronounced intermediate-ratio
failure described below; they should not be interpreted as a uniformly flat
retention tradeoff across tasks.

#### CWE 8K generation anomaly

CWE 8K is strongly non-monotonic: the 40%/30% collapse is present in the raw
generated answers, not a scoring mismatch. There are 291 zero-score examples at
40% and 244 at 30%, versus zero at full cache and one at 20%. Median output
lengths, retokenized with the cached pinned model tokenizer, are 44 / 8.5 / 21 /
44 tokens for full / 40% / 30% / 20%. Of 301 examples scoring at most 30 at 40%
retention, 256 have at most ten retokenized tokens. A deterministic ten-example
sample (indices 0, 1, 2, 8, 11, 12, 14, 17, 21, 22) contains nine short malformed
fragments and one repetitive output. Widespread output-cap exhaustion therefore
does not explain the collapse; exact generated IDs and stopping reasons were not
saved, so neither EOS stopping nor cap exhaustion is asserted for individual rows.

Actual mean retention is 0.399999 at 40% and 0.300000 at 30%, equal to model
selection, with no protected-window contribution. A read-only code audit found
no concrete cache-reuse, position, or ratio-order defect: the runtime uses a
RetainCache whose mask is replaced from unchanged scores; generation restores
the context cache length and preserves prefix/query tokens. This does not rule
out runtime/kernel effects or establish the cause. The anomaly remains
unexplained; scores are retained unchanged. A fresh-cache/order-swapped replay
would be a follow-up diagnostic, not part of the production result protocol.

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
