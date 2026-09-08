# Uniform checkpoint: production manifest for approval

**Status: awaiting approval; zero production jobs submitted.**

The [pilots passed](ruler-pilot-evidence.md). This is the exact 107-job proposal
for [PR #22](https://github.com/d4nieldev/FastKVzip/pull/22), runtime commit
`7323c0d7f891b1ba2fca9c637ed086d057d24819`. Later documentation-only commits do
not change the frozen runtime. The machine-readable manifest lives locally at
`.slurm/grids/uniform-eval-all-w0/manifest.json`; its canonical approval digest is
`30adbdab4df476c1c05c82a75f911b4172e4c84fd58abac771ee372fd0842818`.

## Shared settings and identity

Each job runs one complete prepared/filtered benchmark, without `--num`:
78 RULER configurations (13 tasks × six lengths), 27 SCBench configurations,
SQuAD, and GSM. There are no Agentic evaluation jobs. Total coverage is
60,070 contexts, 138,483 questions and **830,898 answer generations** (five
compressed answers plus one full-cache answer per question).

- One `rtx_pro_6000:1` GPU, standard wrapper requesting `main` / `normal` QoS.
  The site automatically routes GPU jobs to partition `gpu`, applying
  `gpu-part` partition QoS. It assigns supporting CPUs; no CPU override.
- `--level pair --window-size 0 --ratios 0.75 0.50 0.40 0.30 0.20`
  `--full-cache-answer --existing-results resume --ruler-prompt-mode graphkv`.
- Token/graph microbatches `16000/16`, checkpoint subgraphs `2000`, exact saved
  28-token prefix. Generation revision 2; window revision 1.
- All three checkpoints have SHA-256
  `1f3e7c5354a966aa4409af9e609268e796fbec813e7b7d3c683843d9ba177ad0`.
  Model `Qwen/Qwen2.5-7B-Instruct-1M`, revision
  `e28526f7bb80e2a9c8af03b831a9af3812f18fba`; actual gate/sink 16/16.
- One new evaluation-only W&B destination:
  [hdo2z4x4](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-answer-qwen25-7b1m-s40n40-grid-v1/runs/hdo2z4x4),
  named `q25a-s40n40-n200e2-uniform-s0-eval-all-w0`. Source training run
  `61ewyfcm` is provenance only. Workers bind `--wandb-run-id hdo2z4x4`, omit
  `--log-to-wandb`, and never modify checkpoint metadata.
- One serial coordinator retrieves completed results and invokes existing
  `results.coordinator` completeness/conflict/idempotent-upload checks. Keep
  ordinary `test/<task>` keys and retention axis; upload each completed benchmark
  promptly rather than waiting for the entire grid.

Checkpoints/results remain in durable per-account home storage. Standard HF
caching is used; no scratch reservation, teacher-answer cache, or cache-warming
job is needed. RULER revisions are pinned in code; observed source revisions
for every benchmark family are also recorded in the manifest and W&B config.

## Accounts and resolved paths

For each account, `CHECKPOINT` is
`PROJECT/graph_checkpoints/answer/q25a-s40n40-n200e2-uniform-s0/best.pt`.
`RUN_NAME=uniform-eval-all-w0-DATA`; `RUN_DIR=PROJECT/results/RUN_NAME`.
Every benchmark has a unique directory and at most one active writer.

| Account / SSH alias | PROJECT | VENV | Jobs | Estimated GPU-hours |
| --- | --- | --- | ---: | ---: |
| danieloh / `bgu-slurm` | `/home/danieloh/FastKVzip-ruler-evaluation` | `/home/danieloh/.venvs/fastkvzip` | 36 | 154.9 |
| guyzagor / `bgu-slurm-guyzagor` | `/home/guyzagor/danieloh/FastKVzip-ruler-evaluation` | `/home/guyzagor/danieloh/.venvs/fastkvzip` | 37 | 155.4 |
| odedshah / `bgu-slurm-odedshah` | `/home/odedshah/FastKVzip-ruler-evaluation` | `/home/odedshah/FastKVzip/.venv` | 34 | 181.5 |

Assignments use longest-estimated-job first within each tier, placing work on
the least-loaded modeled account slot, not equal job counts. The indivisible
SQuAD job makes odedshah's total larger. Planning uses five slots per account
(15 total). Live inspection of the effective GPU partition confirms
`QoS=gpu-part` and `MaxTRESPU=gres/gpu=5`: five GPUs per user. Normal user QoS
does not override this partition limit. Slurm availability and unrelated work
remain authoritative; no policy overrides or completion dependencies are added
to force concurrency.

Live checks on 2026-09-08 confirmed all three clean runtime commits, identical
checkpoint hashes, allocation-verified matching environments, writable result
paths and 167T free on the shared home filesystem. Only the three existing
CPU dashboard-agent jobs were running for these accounts; no unrelated GPU jobs
were present. Slurm 25.11.4 reported the effective GPU partition up, a seven-day
maximum and 21 free RTX PRO 6000 GPUs. Non-submitting `sbatch --test-only`
checks passed on all three accounts, including the 52G SCBench KV request and
112.25-hour SQuAD request. No jobs were created by these checks. Their start-time
predictions extend into later weeks and are not guarantees; queue priority and
backfill may dominate elapsed calendar time. This is a snapshot, not a
reservation; recheck before submit.

## Resource evidence and uncertainty

The central projection is **491.8 GPU-hours**, not a promised completion time.
Requested wall-time envelopes total 915.25 GPU-hours; jobs release resources
when they finish. Host-memory requests range from 10G to 52G.

The estimates combine complete dataset/query counts and measured token-length
distributions with corrected 4K pilots, three GraphKV 128K pilots and the
15-question SQuAD timing check. Per-context prefill/scoring cost is modeled
separately from six generations per question. Requests round up to 15 minutes,
using 1.75× runtime headroom for RULER and 2× for other tasks; host RAM includes
full-dataset/result overhead plus 30% and 1 GiB headroom.

SQuAD has **87,599 questions**, not 100 contexts: estimate 56.1 hours, request
`4-16:15:00` and 12G. It is the likely long tail even with 15 simultaneous jobs.
Its first 15 questions are a small nonrandom sample. SCBench/GSM response-length
forecasts are assumptions constrained by existing task caps; their uncertainty
is larger than the measured RULER cases. Cold downloads, I/O and contention may
increase runtime. If a job times out, retain its identity and compatible partial
outputs and investigate before a separately approved retry.

The prior full SCBench KV job `21061591` took 4h11m at window 0.02 and reached
42.5 GiB allocated GPU memory. It is sizing evidence, **not an exact compressed
score regression target** for window 0. The new KV estimate is 6.0 hours with a
12-hour / 52G request. All jobs retain the tested 96GB-class GPU and microbatches.
No 48GB-card equivalence is assumed.

Detailed equations, per-task assumptions, sensitivity scenarios, source hashes
and measured logs are preserved under the ignored grid directory in
`resource-estimates.json`, `dataset-inventory.json`, `scbench-lengths.json` and
`pilot-validation.json`.

## Exact submission order and requests

Rows 1, 2–14, 15–27 and 28–107 are submission tiers: SCBench KV, RULER 4K,
RULER 8K, then the rest. Capture all job IDs in one tier before submitting the
next, **without waiting for completion**. Slurm can still start jobs out of order.
The expired training dependency `afterok:21061828` is omitted under the user's
explicit exception; its completed job ID remains provenance.

Wall time is Slurm `D-HH:MM:SS` / `HH:MM:SS`; memory is host RAM, not GPU memory.
Every row requests the single GPU and common settings above.

| # | Benchmark | Account | Contexts | Questions | Time | RAM | Est. GPU-h |
| ---: | --- | --- | ---: | ---: | --- | ---: | ---: |
| 1 | `scbench_kv` | danieloh | 100 | 500 | 12:00:00 | 52G | 6.00 |
| 2 | `ruler_niah_multiquery_4k` | guyzagor | 500 | 500 | 02:45:00 | 10G | 1.50 |
| 3 | `ruler_cwe_4k` | odedshah | 500 | 500 | 01:45:00 | 10G | 0.97 |
| 4 | `ruler_niah_multivalue_4k` | odedshah | 500 | 500 | 01:45:00 | 10G | 0.95 |
| 5 | `ruler_niah_single_3_4k` | guyzagor | 500 | 500 | 01:15:00 | 10G | 0.71 |
| 6 | `ruler_niah_multikey_3_4k` | odedshah | 500 | 500 | 01:15:00 | 10G | 0.67 |
| 7 | `ruler_vt_4k` | guyzagor | 500 | 500 | 01:00:00 | 10G | 0.45 |
| 8 | `ruler_fwe_4k` | odedshah | 500 | 500 | 00:45:00 | 10G | 0.39 |
| 9 | `ruler_niah_single_1_4k` | guyzagor | 500 | 500 | 00:45:00 | 10G | 0.35 |
| 10 | `ruler_niah_single_2_4k` | odedshah | 500 | 500 | 00:45:00 | 10G | 0.35 |
| 11 | `ruler_niah_multikey_2_4k` | guyzagor | 500 | 500 | 00:45:00 | 10G | 0.34 |
| 12 | `ruler_niah_multikey_1_4k` | danieloh | 500 | 500 | 00:45:00 | 10G | 0.34 |
| 13 | `ruler_qa_2_4k` | danieloh | 500 | 500 | 00:30:00 | 10G | 0.19 |
| 14 | `ruler_qa_1_4k` | danieloh | 500 | 500 | 00:30:00 | 10G | 0.16 |
| 15 | `ruler_niah_multiquery_8k` | danieloh | 500 | 500 | 03:00:00 | 10G | 1.72 |
| 16 | `ruler_cwe_8k` | danieloh | 500 | 500 | 02:15:00 | 10G | 1.19 |
| 17 | `ruler_niah_multivalue_8k` | danieloh | 500 | 500 | 02:00:00 | 10G | 1.15 |
| 18 | `ruler_niah_multikey_3_8k` | danieloh | 500 | 500 | 01:45:00 | 10G | 0.95 |
| 19 | `ruler_niah_single_3_8k` | guyzagor | 500 | 500 | 01:45:00 | 10G | 0.90 |
| 20 | `ruler_vt_8k` | odedshah | 500 | 500 | 01:15:00 | 10G | 0.65 |
| 21 | `ruler_fwe_8k` | guyzagor | 500 | 500 | 01:15:00 | 10G | 0.58 |
| 22 | `ruler_niah_single_1_8k` | odedshah | 500 | 500 | 01:00:00 | 10G | 0.55 |
| 23 | `ruler_niah_single_2_8k` | guyzagor | 500 | 500 | 01:00:00 | 10G | 0.52 |
| 24 | `ruler_niah_multikey_2_8k` | odedshah | 500 | 500 | 01:00:00 | 10G | 0.52 |
| 25 | `ruler_niah_multikey_1_8k` | guyzagor | 500 | 500 | 01:00:00 | 10G | 0.51 |
| 26 | `ruler_qa_2_8k` | guyzagor | 500 | 500 | 00:45:00 | 10G | 0.36 |
| 27 | `ruler_qa_1_8k` | odedshah | 500 | 500 | 00:45:00 | 10G | 0.32 |
| 28 | `squad` | odedshah | 18,891 | 87,599 | 4-16:15:00 | 12G | 56.12 |
| 29 | `ruler_niah_multiquery_128k` | odedshah | 500 | 500 | 1-08:00:00 | 44G | 18.17 |
| 30 | `scbench_repoqa_and_kv` | guyzagor | 88 | 704 | 1-11:00:00 | 28G | 17.26 |
| 31 | `ruler_cwe_128k` | odedshah | 500 | 500 | 1-05:45:00 | 44G | 16.95 |
| 32 | `ruler_niah_multivalue_128k` | odedshah | 500 | 500 | 1-05:45:00 | 44G | 16.94 |
| 33 | `ruler_niah_single_3_128k` | guyzagor | 500 | 500 | 1-04:45:00 | 44G | 16.40 |
| 34 | `ruler_niah_multikey_3_128k` | guyzagor | 500 | 500 | 1-04:30:00 | 42G | 16.23 |
| 35 | `ruler_vt_128k` | odedshah | 500 | 500 | 1-03:45:00 | 44G | 15.78 |
| 36 | `ruler_niah_single_1_128k` | guyzagor | 500 | 500 | 1-03:30:00 | 44G | 15.62 |
| 37 | `ruler_niah_single_2_128k` | danieloh | 500 | 500 | 1-03:15:00 | 44G | 15.57 |
| 38 | `ruler_niah_multikey_1_128k` | danieloh | 500 | 500 | 1-03:15:00 | 44G | 15.55 |
| 39 | `ruler_niah_multikey_2_128k` | danieloh | 500 | 500 | 1-03:15:00 | 44G | 15.51 |
| 40 | `ruler_qa_2_128k` | guyzagor | 500 | 500 | 1-03:00:00 | 44G | 14.81 |
| 41 | `ruler_qa_1_128k` | danieloh | 500 | 500 | 1-02:45:00 | 44G | 14.70 |
| 42 | `ruler_fwe_128k` | danieloh | 500 | 500 | 1-03:30:00 | 44G | 14.27 |
| 43 | `scbench_repoqa` | guyzagor | 88 | 440 | 22:15:00 | 28G | 10.96 |
| 44 | `ruler_niah_multiquery_64k` | danieloh | 500 | 500 | 13:45:00 | 26G | 7.83 |
| 45 | `ruler_cwe_64k` | danieloh | 500 | 500 | 12:30:00 | 26G | 7.01 |
| 46 | `ruler_niah_multivalue_64k` | danieloh | 500 | 500 | 12:15:00 | 26G | 6.95 |
| 47 | `ruler_niah_multikey_3_64k` | danieloh | 500 | 500 | 11:45:00 | 26G | 6.60 |
| 48 | `ruler_niah_single_3_64k` | guyzagor | 500 | 500 | 11:30:00 | 26G | 6.57 |
| 49 | `scbench_summary` | odedshah | 70 | 350 | 13:30:00 | 38G | 6.47 |
| 50 | `ruler_vt_64k` | guyzagor | 500 | 500 | 11:00:00 | 26G | 6.18 |
| 51 | `ruler_niah_single_1_64k` | guyzagor | 500 | 500 | 10:30:00 | 26G | 6.00 |
| 52 | `ruler_niah_single_2_64k` | odedshah | 500 | 500 | 10:30:00 | 26G | 5.98 |
| 53 | `ruler_niah_multikey_2_64k` | odedshah | 500 | 500 | 10:30:00 | 26G | 5.98 |
| 54 | `ruler_niah_multikey_1_64k` | guyzagor | 500 | 500 | 10:30:00 | 26G | 5.97 |
| 55 | `ruler_fwe_64k` | odedshah | 500 | 500 | 10:45:00 | 26G | 5.79 |
| 56 | `ruler_qa_2_64k` | danieloh | 500 | 500 | 10:30:00 | 26G | 5.64 |
| 57 | `ruler_qa_1_64k` | guyzagor | 500 | 500 | 10:15:00 | 26G | 5.54 |
| 58 | `scbench_mf` | danieloh | 100 | 600 | 10:00:00 | 48G | 4.81 |
| 59 | `ruler_niah_multiquery_32k` | odedshah | 500 | 500 | 07:00:00 | 16G | 3.90 |
| 60 | `scbench_kv_mid` | guyzagor | 100 | 1,000 | 07:45:00 | 30G | 3.88 |
| 61 | `scbench_vt` | guyzagor | 90 | 450 | 07:45:00 | 40G | 3.79 |
| 62 | `scbench_prefix_suffix` | danieloh | 100 | 500 | 07:30:00 | 38G | 3.69 |
| 63 | `ruler_cwe_32k` | danieloh | 500 | 500 | 05:45:00 | 16G | 3.22 |
| 64 | `ruler_niah_multivalue_32k` | odedshah | 500 | 500 | 05:45:00 | 16G | 3.20 |
| 65 | `scbench_summary_with_needles` | odedshah | 70 | 560 | 06:45:00 | 38G | 3.07 |
| 66 | `ruler_niah_multikey_3_32k` | guyzagor | 500 | 500 | 05:15:00 | 16G | 2.94 |
| 67 | `ruler_niah_single_3_32k` | danieloh | 500 | 500 | 05:15:00 | 16G | 2.89 |
| 68 | `ruler_vt_32k` | odedshah | 500 | 500 | 04:30:00 | 16G | 2.57 |
| 69 | `ruler_niah_single_1_32k` | danieloh | 500 | 500 | 04:15:00 | 16G | 2.44 |
| 70 | `ruler_niah_single_2_32k` | danieloh | 500 | 500 | 04:15:00 | 16G | 2.42 |
| 71 | `ruler_niah_multikey_1_32k` | odedshah | 500 | 500 | 04:15:00 | 16G | 2.41 |
| 72 | `ruler_niah_multikey_2_32k` | danieloh | 500 | 500 | 04:15:00 | 16G | 2.39 |
| 73 | `ruler_fwe_32k` | guyzagor | 500 | 500 | 04:30:00 | 16G | 2.37 |
| 74 | `ruler_niah_multiquery_16k` | odedshah | 500 | 500 | 04:15:00 | 12G | 2.35 |
| 75 | `scbench_mf_mid` | guyzagor | 100 | 600 | 04:45:00 | 28G | 2.25 |
| 76 | `ruler_qa_1_32k` | odedshah | 500 | 500 | 04:00:00 | 16G | 2.15 |
| 77 | `ruler_qa_2_32k` | guyzagor | 500 | 500 | 04:00:00 | 16G | 2.12 |
| 78 | `scbench_prefix_suffix_mid` | guyzagor | 100 | 500 | 03:45:00 | 22G | 1.85 |
| 79 | `scbench_kv_short` | odedshah | 100 | 1,000 | 03:45:00 | 14G | 1.80 |
| 80 | `scbench_repoqa_short` | danieloh | 22 | 110 | 03:45:00 | 14G | 1.77 |
| 81 | `ruler_cwe_16k` | danieloh | 500 | 500 | 03:15:00 | 12G | 1.75 |
| 82 | `ruler_niah_multivalue_16k` | danieloh | 500 | 500 | 03:00:00 | 12G | 1.73 |
| 83 | `ruler_niah_single_3_16k` | guyzagor | 500 | 500 | 02:45:00 | 12G | 1.47 |
| 84 | `ruler_niah_multikey_3_16k` | danieloh | 500 | 500 | 02:45:00 | 12G | 1.46 |
| 85 | `scbench_kv_tiny` | odedshah | 100 | 1,000 | 03:00:00 | 12G | 1.44 |
| 86 | `gsm` | guyzagor | 148 | 148 | 03:00:00 | 10G | 1.42 |
| 87 | `ruler_vt_16k` | danieloh | 500 | 500 | 02:15:00 | 12G | 1.18 |
| 88 | `scbench_mf_short` | guyzagor | 100 | 600 | 02:15:00 | 16G | 1.07 |
| 89 | `ruler_niah_single_2_16k` | guyzagor | 500 | 500 | 02:00:00 | 12G | 1.06 |
| 90 | `ruler_niah_multikey_1_16k` | odedshah | 500 | 500 | 02:00:00 | 12G | 1.05 |
| 91 | `ruler_niah_single_1_16k` | danieloh | 500 | 500 | 02:00:00 | 12G | 1.04 |
| 92 | `ruler_niah_multikey_2_16k` | odedshah | 500 | 500 | 02:00:00 | 12G | 1.04 |
| 93 | `ruler_fwe_16k` | guyzagor | 500 | 500 | 02:00:00 | 12G | 1.03 |
| 94 | `scbench_prefix_suffix_short` | odedshah | 100 | 500 | 02:00:00 | 14G | 0.91 |
| 95 | `ruler_qa_2_16k` | danieloh | 500 | 500 | 01:45:00 | 12G | 0.85 |
| 96 | `ruler_qa_1_16k` | danieloh | 500 | 500 | 01:45:00 | 12G | 0.74 |
| 97 | `scbench_mf_tiny` | odedshah | 100 | 600 | 01:30:00 | 12G | 0.72 |
| 98 | `scbench_prefix_suffix_tiny` | guyzagor | 100 | 500 | 01:30:00 | 10G | 0.71 |
| 99 | `scbench_repoqa_tiny` | guyzagor | 9 | 45 | 01:30:00 | 10G | 0.70 |
| 100 | `scbench_summary_tiny` | danieloh | 70 | 70 | 01:30:00 | 14G | 0.65 |
| 101 | `scbench_summary_mid` | odedshah | 62 | 62 | 01:15:00 | 14G | 0.60 |
| 102 | `scbench_summary_short` | odedshah | 62 | 62 | 01:15:00 | 14G | 0.60 |
| 103 | `scbench_qa_eng` | guyzagor | 20 | 102 | 01:30:00 | 40G | 0.58 |
| 104 | `scbench_choice_eng` | guyzagor | 18 | 71 | 01:15:00 | 40G | 0.44 |
| 105 | `scbench_many_shot` | danieloh | 54 | 270 | 00:45:00 | 16G | 0.31 |
| 106 | `scbench_many_shot_short` | danieloh | 54 | 270 | 00:30:00 | 14G | 0.23 |
| 107 | `scbench_many_shot_tiny` | odedshah | 54 | 270 | 00:30:00 | 10G | 0.18 |

## Exact command construction

Resolve uppercase placeholders from the account table and the numbered row;
these are the same arguments stored as arrays in the machine-readable manifest.
The existing batch script supplies `--partition=main --ntasks=1` and the durable
`--run-dir`. All paths are absolute. Submit only after this manifest is approved.

```bash
ssh ALIAS sbatch --parsable \
  --chdir=PROJECT --job-name=RUN_NAME \
  --output=PROJECT/.slurm/logs/%j-%x.log \
  --gpus=rtx_pro_6000:1 --time=TIME --mem=MEM \
  --export=ALL,FASTKVZIP_VENV=VENV,EVAL_GRAPH_SCRIPT=prefill/eval_graph.py \
  PROJECT/slurm/eval_graph.sbatch RUN_NAME \
  --graph-checkpoint CHECKPOINT --data DATA \
  --level pair --window-size 0 --ratios 0.75 0.50 0.40 0.30 0.20 \
  --full-cache-answer --existing-results resume \
  --token-microbatch-size 16000 --graph-microbatch-size 16 \
  --ruler-prompt-mode graphkv --wandb-run-id hdo2z4x4
```

Before each scheduler mutation, append a `submitting` attempt to the durable
receipt; then persist the exact returned job ID. Stop all new submissions on
an unknown outcome and reconcile, never blindly retry. Preserve every attempt.
Never resume the superseded pilot outputs into these production directories.

## Approval gates remaining

1. Approve this exact manifest, assignments, full dataset sizes and resources.
2. Permit sending job IDs and the experiment name to the existing
   `https://graphfastkvzip.onrender.com` dashboard. The authorization reviewer
   blocked that export, so pilot IDs have not been attached yet. No checkpoint,
   model output or credential is included. Use the existing project
   `graphkv-answer-qwen25-7b1m-s40n40-grid-v1`, retaining all 12 pilot attempts
   and later production attempts.

No production submission is authorized by the successful pilots, the new W&B
destination, or this document alone.
