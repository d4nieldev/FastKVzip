# RULER pilot evidence — 2026-09-08

These are **one-example-per-task smoke tests, not benchmark results**: the first
example from all 13 RULER 4K tasks and three 128K tasks, in both prompt modes.
Every RULER snapshot correctly remains incomplete at **1/500** examples. A
separate SQuAD timing check covers three contexts, not the complete benchmark.
All pilot jobs disabled W&B (`WANDB_MODE=disabled`); no pilot metrics were
uploaded or used to claim production benchmark completion.

## Checkpoint and protocol

- Checkpoint: `graph_checkpoints/answer/q25a-s40n40-n200e2-uniform-s0/best.pt`;
  source W&B run `61ewyfcm`; 1,089,385,068 bytes; SHA-256
  `1f3e7c5354a966aa4409af9e609268e796fbec813e7b7d3c683843d9ba177ad0`.
- Actual metadata: `Qwen/Qwen2.5-7B-Instruct-1M`, gate dimension/sink keys
  **16/16**, graph dimension 32, BF16, 2,000-token subgraphs. The `s40n40` name
  denotes the source data range, not gate dimensions.
- Standard `slurm/eval_graph.sbatch` / `prefill/eval_graph.py`, `--level pair`,
  `--window-size 0`, ratios `0.75 0.50 0.40 0.30 0.20`, full-cache answers,
  token/graph microbatches `16000/16`, and `--existing-results resume`.
  RULER uses `--num 1`; SQuAD uses `--num 3`.
- GraphKV restores the checkpoint's exact 28-token prefix (digest
  `bf4d3f4005b6ad64772f7ebf21b7b3e6520e49641831fb2138a4984b540364fc`)
  and ordinary `Q:` query path. Official mode omits the extra GraphKV general
  instruction and `Q:` prefix, using the existing model chat boundaries.
  Published demonstrations remain in context; the final question and completion
  cue stay together outside the compression budget. This is adapted prompt
  evaluation, not a claim of bitwise upstream prompt reproduction.
- Pinned HF revisions: 4K `90daf679d2893abc90bbc9451f2a1f33de86c66e`;
  128K `5bdc2f0e2a6e2dc79abbc65378b504d400397415`.
  PyTorch `2.7.0+cu128`, Transformers `4.51.3`, Datasets `4.0.0`, and
  FlashAttention `2.7.3` match across the three accounts.

## Corrected window-zero evidence

The original 4K jobs `21112931` / `21112932` and SQuAD job `21112942` exposed a
pre-existing bug: integer zero became a protected 2% tail when context length
was below the prefill chunk. Those three attempts are **invalid as window-zero
evidence** and are excluded from the tables below. Their attempt IDs/logs remain
history; corrected jobs are `21113328`, `21113329`, and `21113330`.

The correction is commit `7323c0d7f891b1ba2fca9c637ed086d057d24819`.
Corrected manifests have `window_revision: 1`; all 26 4K task logs and all three
SQuAD context logs report `Local window 0`. For every saved ratio,
`actual_retention == model_selection_rate` at stored precision. Small deviations
from the requested ratio, such as 0.2999 for 0.30, are retained in diagnostics.

The six 128K jobs used clean commit
`78d5d404067d046efad8bb5007f41050b2701d9f` and already had zero protected tokens:
their contexts exceed the 16K prefill chunk. They remain valid standalone pilot
evidence. Their old manifests load as window revision 0 and cannot be resumed
or uploaded through the corrected production coordinator. Generation revision
is 2 in both sets; old artifacts were not relabeled as corrected executions.

## Exact pilot scores

Scores are absolute percentages against the published reference lists:
case-insensitive substring recall for retrieval/tracing/frequency tasks, and
any-reference inclusion for QA. `Full` means the unpruned full-cache answer;
the remaining columns are requested retention percentages, not compression
percentages. No relative-to-full score is used here.

### 4K — corrected runs

| Task | Mode | Full | 75 | 50 | 40 | 30 | 20 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| niah_single_1 | graphkv | 100 | 100 | 100 | 100 | 100 | 100 |
| niah_single_1 | official | 100 | 100 | 100 | 100 | 100 | 100 |
| niah_single_2 | graphkv | 100 | 100 | 100 | 100 | 100 | 100 |
| niah_single_2 | official | 100 | 100 | 100 | 100 | 100 | 100 |
| niah_single_3 | graphkv | 100 | 100 | 100 | 100 | 100 | 100 |
| niah_single_3 | official | 100 | 100 | 100 | 100 | 100 | 100 |
| niah_multikey_1 | graphkv | 100 | 100 | 100 | 100 | 100 | 100 |
| niah_multikey_1 | official | 100 | 100 | 100 | 100 | 100 | 100 |
| niah_multikey_2 | graphkv | 100 | 100 | 100 | 100 | 100 | 100 |
| niah_multikey_2 | official | 100 | 100 | 100 | 100 | 100 | 100 |
| niah_multikey_3 | graphkv | 100 | 100 | 100 | 100 | 100 | 100 |
| niah_multikey_3 | official | 100 | 100 | 100 | 100 | 100 | 100 |
| niah_multivalue | graphkv | 75 | 75 | 75 | 75 | 75 | 75 |
| niah_multivalue | official | 100 | 100 | 100 | 100 | 100 | 100 |
| niah_multiquery | graphkv | 100 | 100 | 100 | 100 | 100 | 100 |
| niah_multiquery | official | 100 | 100 | 100 | 100 | 100 | 100 |
| vt | graphkv | 100 | 100 | 100 | 100 | 100 | 100 |
| vt | official | 100 | 100 | 0 | 0 | 0 | 0 |
| cwe | graphkv | 100 | 100 | 100 | 100 | 100 | 100 |
| cwe | official | 90 | 90 | 90 | 90 | 90 | 100 |
| fwe | graphkv | 100 | 100 | 100 | 100 | 100 | 100 |
| fwe | official | 100 | 100 | 100 | 100 | 100 | 100 |
| qa_1 | graphkv | 100 | 100 | 100 | 100 | 100 | 100 |
| qa_1 | official | 100 | 100 | 100 | 100 | 100 | 100 |
| qa_2 | graphkv | 100 | 100 | 100 | 100 | 100 | 100 |
| qa_2 | official | 100 | 100 | 100 | 100 | 100 | 100 |

### 128K — unaffected long-context runs

| Task | Mode | Full | 75 | 50 | 40 | 30 | 20 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| niah_multikey_3 | graphkv | 100 | 100 | 100 | 100 | 100 | 100 |
| niah_multikey_3 | official | 100 | 100 | 100 | 100 | 100 | 100 |
| vt | graphkv | 100 | 100 | 60 | 60 | 80 | 40 |
| vt | official | 0 | 0 | 0 | 0 | 0 | 0 |
| qa_2 | graphkv | 100 | 0 | 0 | 100 | 100 | 0 |
| qa_2 | official | 0 | 0 | 100 | 100 | 100 | 0 |

## Output and prompt inspection

All 13 pinned first-row 4K questions/reference lists were inspected without
dumping contexts. All 26 corrected task/mode examples and 130 pruned score
records were checked against them; scores independently recomputed from output
text match `metrics.json`. Each example preserves identical references and
full-cache answers across ratios. The six 128K output/metric pairs also agree.

- **4K multivalue:** GraphKV omits `5107245` from four required numbers even at
  full cache, explaining 75 rather than a pruning-induced failure. Official
  retrieves all four at every corrected ratio.
- **4K VT:** the question asks for variables assigned `15311`; the targets are
  `FITJT VGCAO ZJQUQ TYFAD DROFS`. Official outputs at 50–20% repeat the
  completion cue and exhaust exactly 30 tokenizer tokens before naming a
  target. The capped final token is preserved; this is not the old unconditional
  final-token truncation bug. GraphKV returns all targets at every ratio.
- **4K CWE:** official full-cache output repeats `speaking` and omits `abrasive`,
  correctly scoring 90. Its 20% output contains all ten distinct targets (100).
- **128K VT:** GraphKV 20% returns `VAR SSBEA SSWRJ`, matching two of five
  references (40). Official outputs spend the cap on an explanation/cue and
  match no reference, including the full-cache output.
- **128K QA2:** the reference is `yes`. Official full cache actually answers
  **`no` and scores 0**, not 100. GraphKV full cache answers `yes`; the pruned
  answers vary non-monotonically with retention. A single binary question
  cannot establish that one retention ratio or mode is generally better.

Compared with the invalid short-context attempts, corrected GraphKV scores
are unchanged; official 20% multivalue improves 75→100 and CWE 90→100, while
official VT at 50% changes 100→0. These are observations about the sampled
outputs, not statistically meaningful comparisons.

## Runtime and memory evidence

Every job below exited with application status 0 on one RTX Pro 6000 (95.0 GiB
reported device capacity). The six 128K jobs requested 60G host memory; the
corrected 4K/SQuAD jobs requested 12G after measuring the initial attempts.
Task work is the
sum of logged prefill/scoring/generation times (rounded to 0.1s); wall time is
the standard wrapper measured by `/usr/bin/time -v`, including startup/loading.
GPU peak is **PyTorch allocated memory**, not total device usage. RSS is the
process peak from GNU time, converted from KiB to GiB; it is not per-task RSS.
These measurements do not justify a full-suite runtime extrapolation by
themselves.

| Job | Account | Dataset / mode | Task work (s) | Wall time | Peak GPU (GiB) | Peak RSS (GiB) |
| --- | --- | --- | ---: | --- | ---: | ---: |
| 21113328 | danieloh | all 13 4K / graphkv | 47.6 | 1:27.98 | 17.1 | 5.58 |
| 21113329 | guyzagor | all 13 4K / official | 70.6 | 1:49.67 | 17.1 | 5.58 |
| 21112933 | odedshah | niah_multikey_3 128K / graphkv | 120.0 | 3:27.66 | 40.4 | 29.62 |
| 21112935 | danieloh | niah_multikey_3 128K / official | 61.3 | 1:47.53 | 40.4 | 29.11 |
| 21112936 | guyzagor | vt 128K / graphkv | 119.1 | 2:23.57 | 40.5 | 30.40 |
| 21112937 | odedshah | vt 128K / official | 91.7 | 2:49.75 | 40.5 | 30.81 |
| 21112938 | danieloh | qa_2 128K / graphkv | 81.4 | 1:45.71 | 40.3 | 29.51 |
| 21112939 | guyzagor | qa_2 128K / official | 82.0 | 1:54.63 | 40.3 | 29.44 |
| 21113330 | odedshah | three SQuAD contexts / graphkv | 34.5 | 0:54.19 | 14.9 | 5.71 |

The SQuAD check contains 15 questions and 75 pruned records. All have zero
protected-window contribution; its snapshot correctly remains incomplete at
3/18,891 contexts.

## Evidence locations

Local execution artifacts are intentionally ignored, under
`.slurm/grids/uniform-eval-all-w0/pilots/<account>/`:
`logs/<job>-<run>.log`, `preflight/<run>-preflight.json`, and
`results/<run>/{manifest.json,datasets.json,metrics.json,outputs/<task>/0.json}`.
Run names are `uniform-pilot-<data>-<mode>-w0`; official RULER result task names
append `_official`. The tables above transcribe those saved artifacts, not W&B
history. The [approved plan](plans/ruler-evaluation.md) remains the production
scope; completing these pilots does not complete that plan's benchmark runs.
