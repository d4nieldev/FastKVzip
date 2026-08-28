# Running implicit graph experiments

This guide uses the implicit whole-context mixer in this PR.
Run commands from the repository root on the BGU cluster.

## One-time setup

```bash
cd /home/danieloh/FastKVzip-implicit
git switch main
git pull --ff-only
mkdir -p .slurm/logs
export FASTKVZIP_VENV=/home/danieloh/.venvs/fastkvzip
export MODEL_ID=Qwen/Qwen3-8B
uv pip install --python "$FASTKVZIP_VENV/bin/python" --no-deps datasets==4.0.0
```

The batch scripts activate this existing environment. They do not rebuild
FlashAttention. W&B is online by default. Do not put its API key in a script.
The small `datasets` upgrade is required once. The evaluation script checks
the version before requesting a model or dataset.

## Choose a GPU

Run `sres` immediately before every submission.

Prefer `rtx_pro_6000:1`. Use `rtx_6000:1` when it has better availability.
The GPU type belongs on the `sbatch` command line, not inside a script.

## Train one pilot

Use a new run name for every architecture and seed.

```bash
sres
sbatch --gpus=rtx_pro_6000:1 slurm/train_graph.sbatch gd32-seed0-pilot \
  --gate-checkpoint fastkvzip \
  --graph-dim 32 \
  --seed 0 \
  --max-contexts 1
```

This uses the tested pilot request: one GPU, one hour, 60G RAM, and 40G
scratch. The script writes checkpoints and W&B files to
`runs/graph/<run-name>/`. It writes the teacher cache to node-local scratch.
That cache disappears when the job ends.

Set `TEACHER_CACHE_DIR` only when you deliberately want a durable cache. Give
each concurrent job a different cache directory.

`steps` means one completed training context. It does not mean a gate token
slice or one raw optimizer call. By default, `last.pt` and validation run every
epoch. A validation sweep never writes
`last.pt` after each held-out context. `best.pt` is written when its mean BCE
improves. Pass `--no-save-best` when one `last.pt` is the only desired
checkpoint. Use `last.pt` for a pilot or resume. Prefer `best.pt` for final
evaluation when it exists.

When saving and validation are both due at one boundary, validation runs first
and then `last.pt` is written. This preserves any plateau-scheduler update.

## Change an experiment

Pass only the options you want to change after the run name.

| Choice | Option |
|---|---|
| mixer width | `--graph-dim` |
| Gram scale | `--gram-normalization token-count` or `none` |
| activation slope | `--leaky-relu-slope` |
| residual start | `--alpha-init` |
| training schedule | `--training-mode joint` or `two-phase` |
| learning rates | `--gate-lr`, `--mixer-lr` |
| run length | `--epochs`, `--max-contexts` |
| checkpoint cadence | `--save-strategy`, `--save-every` |
| best checkpoint | `--save-best` or `--no-save-best` |
| validation cadence | `--eval-strategy`, `--eval-every` |
| memory/speed knobs | `--graph-microbatch-size`, `--token-microbatch-size` |

Joint mode is the default. Two-phase mode updates the gate in token slices,
then updates the mixer once per context. Scheduler arguments must be JSON.
Both schedulers default to `none`.
`--max-contexts` also counts training contexts only.

Use `steps` for a number of completed training contexts. Use `epochs` for
completed training epochs. The defaults are `--save-strategy epochs
--save-every 1` and `--eval-strategy epochs --eval-every 1`.

In joint mode, token microbatch size changes memory and speed only. In
two-phase mode, it also changes the gate update batch and update count.

For more throughput, increase the token microbatch first. It does more work
per GPU call and uses more GPU memory. Then increase the graph microbatch if
memory remains. That runs more complete layer/head graphs in parallel and also
uses more memory. Gate projection and scoring are batched across the graph
microbatch. Only RMSNorm is grouped by transformer layer.

```bash
--gate-lr-scheduler StepLR --gate-lr-scheduler-kwargs '{"step_size": 1, "gamma": 0.5}'
```

Do not reuse a run name. Do not let two jobs fill the same missing shared
teacher cache. The default per-job scratch cache avoids that race.

When every expected teacher-cache file exists, training unloads the base LLM
after it constructs the student. This gives its GPU memory back to training.
If a cache file is later missing, training reloads the base LLM only when that
file is needed.

## Continue a run

Do not pass `--gate-checkpoint` with `--resume`.

```bash
sres
sbatch --gpus=rtx_pro_6000:1 slurm/train_graph.sbatch gd32-seed0 \
  --resume runs/graph/gd32-seed0-pilot/last.pt
```

The checkpoint restores the model settings, prefix, prefill chunk, optimizer,
scheduler, cursor, and W&B run. Architecture and optimizer configuration must
match. You may change save/evaluation cadence when resuming, but keep it fixed
inside a comparable experiment.

For a longer run, first inspect the pilot with `sacct`. Then request measured
time and memory values before the script path in the `sbatch` command. Use
`--tmp=600G` only after rechecking live limits.

## Evaluate a checkpoint

Start with one example. Evaluation runs generation at five pruning ratios.
The environment must use `datasets==4.0.0`, as pinned in
`prefill/requirements.txt`.

```bash
sres
bash slurm/submit_eval_graph.sh gd32-seed0-squad \
  --gpu rtx_pro_6000:1 \
  --time 01:00:00 \
  --mem 60G \
  --graph-checkpoint graph_checkpoints/gd32-seed0/best.pt \
  --data squad \
  --idx 0 \
  --num 1
```

The helper accepts the run name once. It uses that name for the Slurm job, log,
and result directory. It resolves a relative checkpoint from the project root.
It forwards microbatch, ratio, data, and verbosity options to `eval_graph.py`.
By default, an existing run name fails.

Resume a compatible run explicitly:

```bash
bash slurm/submit_eval_graph.sh gd32-seed0-squad \
  --gpu rtx_pro_6000:1 \
  --time 01:00:00 \
  --mem 60G \
  --graph-checkpoint graph_checkpoints/gd32-seed0/best.pt \
  --existing-results resume \
  --data squad \
  --idx 0 \
  --num 1000000
```

Resume skips every saved task/example/ratio and computes only missing work.
It can add new tasks, ranges, or ratios to the same run. `resume` also creates
the run when it does not exist, which is useful for parallel task submissions.
Use `--existing-results overwrite` only to permanently replace the exact run.

Pass the GPU selected after `sres`. Use measured time and memory for a full
benchmark instead of the one-example values above. `--dry-run` prints the
complete `sbatch` command without submitting it.

The evaluator generates one full-cache reference answer per question by
default. Add `--no-full-cache-answer` when you need only retained-cache
answers. Full-cache answers are additive during resume. Enabling them later
fills missing answers in the selected range. Disabling them never removes or
regenerates answers already saved. Relative metrics remain unavailable until
the task has a complete, nonzero full-cache baseline.

Use `--ratios 0.1 0.2 0.3` to evaluate only selected retention ratios. The
metrics parser derives the saved ratio union from the output files. Omitting
the option preserves the original five ratios.

Evaluation uses the checkpoint's microbatch sizes by default. Override them
with `--token-microbatch-size N` and `--graph-microbatch-size N`. Use `full`
and `all` for one token chunk per context and all layer/head graphs at once.
Their product controls peak scoring memory, so pilot `full` plus `all` before a
long benchmark.

The default loader does not evaluate every upstream SQuAD or GSM8K row.
It stops after adding the 101st unique SQuAD training context because its
condition is `> 100`; that last context contains only the question that caused
the stop. It returns the first 100 GSM8K test examples whose derived context
has at least 72 tokens. SCBench loads every row in each selected preprocessed
split. A large `--num` exhausts these loaded subsets; it does not expand them.

One run uses this layout:

```text
results/<run-name>/
├── manifest.json
├── metrics.json
└── outputs/
    └── <task>/
        └── <example-index>.json
```

The manifest fixes the resolved checkpoint path, the SHA-256 of the complete
checkpoint file, the protected-window size, and the pruning level. Resume
requires all four to match. Tasks, indices, ratios, full-answer coverage, and
microbatch sizes are derived from the output files and may be extended.

Each output keeps the existing answer records and adds `_meta` with its task,
index, dataset size, input fingerprint, and QA keys. Files are replaced
atomically after each complete ratio.

After successful generation, the same job runs the result parser. Readable
metrics appear at the end of the Slurm log. Structured metrics are saved at
`results/<run-name>/metrics.json`. Evaluation or metric failure marks the job
as failed while keeping every valid output already written.

### Log final benchmark metrics to W&B

Pilots and partial benchmarks must not use W&B metric upload. W&B upload
requires every stored task and retention ratio in the run to cover the complete
repo-loaded benchmark. In particular, complete SQuAD coverage is 101 contexts.

For a final run, add:

```bash
--log-to-wandb \
--wandb-project graphkv-e124-g-rand-pre-freeze-tf
```

Add `--wandb-entity ENTITY` only when the training run is under a non-default
entity. The checkpoint supplies the training W&B run ID, and that run must be
finished. The job logs these curves against `test/retention_ratio`:

- `test/<task>`: absolute score.
- `test/<task>-relative`: score relative to that task's full-cache score. This
  curve is omitted until the full-cache baseline is complete and nonzero.
- `test/<task>-actual-retention`: mean achieved retention across examples.

Matching W&B points are skipped. Missing points are added. A different local
and remote value fails without changing W&B. Local outputs remain available
when upload fails. Temporary W&B SDK files use system temporary storage and are
removed when parsing ends, including for the direct retry command below.

Retry parsing or W&B upload without loading the LLM:

```bash
source /home/danieloh/.venvs/fastkvzip/bin/activate
PYTHONPATH="$PWD/prefill" python -m results.parse \
  --run-dir results/gd32-seed0-squad \
  --log-to-wandb \
  --wandb-project graphkv-e124-g-rand-pre-freeze-tf
```

The first number saved for each ratio is the requested retention ratio. The
second is the actual ratio. The actual ratio can be higher when the protected
local window is larger than the requested budget. The protected window is
never partially removed to force a smaller ratio.

### Evaluation progress

Graph evaluation is quiet by default. Each expanded task gets one real
`tqdm` bar. The same line updates during prefill, mixer scoring, and generation,
then advances after the example result is saved. The bar keeps `tqdm`'s normal
count, elapsed time, ETA, and iteration rate. It does not show a separate phase
field.

The postfix shows the current example:

| Field | Meaning |
|---|---|
| `tokens` | Scored context tokens, excluding the protected prefix. |
| `prefill` | Tokenization, transfers, and chunked LLM prefill. |
| `mixer` | Implicit mixer scoring and score assignment. |
| `gen` | Optional full-cache answers, pruning, and all requested-ratio generations. |
| `total` | Time from prefill start through result saving. |
| `gpu` | Peak PyTorch-allocated memory for this example / total GPU memory. |

The active operation displays `...`; later operations display `--`. Timing is
synchronized wall-clock time, so it includes CPU work and CPU/GPU transfers
inside each phase. CUDA peak memory resets for every example.

Per-example QA text, thresholds, nested bars, and other detailed messages are
hidden. Add `--verbose` to restore them. If an example fails in quiet mode, its
captured diagnostics are printed before the traceback. Progress-only data is
not added to the result JSON, and existing answer/result fields are unchanged.

Training W&B logs `validation/bce` once per validation sweep. It is the mean
across the four held-out contexts. It does not log task accuracy or generation
metrics; standalone evaluation writes those benchmark results.

Each completed training context logs `train/bce` (or the two-phase losses),
learning rates, `train/mean_alpha`, `train/epoch`, and
`train/tokens`. The last value counts only scored training context tokens: not
prefix or validation tokens. The terminal has a second `tqdm` bar for these
whole-context training steps; the existing prefill bars remain separate.

## Monitor a job

Save the job ID printed by `sbatch`.

```bash
squeue --jobs=JOB_ID -o '%.18i %.9P %.20j %.2t %.10M %.6D %R'
scontrol show job JOB_ID
tail -f .slurm/logs/JOB_ID-RUN_NAME.log
sacct --jobs=JOB_ID --format=JobID,State,ExitCode,Elapsed,AllocTRES,MaxRSS
```

Use `sstat` only while the job is running. Do not resubmit a pending job.
Read its reason first.

## Submit the fixed Qwen3-8B grid

The fixed grid has nine valid runs: three epoch counts times random/trainable,
pretrained/trainable, and pretrained/frozen gates. A random frozen gate is
invalid for each epoch count.

`slurm/submit_graph_grid.sh` first submits the one-epoch random/trainable run.
It fills the shared cache at
`/groups/ydar_group/danieloh/fastkvzip-implicit/teacher-cache-qwen3-8b-prefill16k`.
It then submits the other eight runs as a throttled array with an `afterok`
dependency. This prevents competing cache writers.

All nine runs use Qwen3-8B, seed 0, default model/training settings, epoch
save/evaluation cadence, `--no-save-best`, and unique directories under
`graph_checkpoints/`.

Run `sres`, choose the GPU type, and use measured full-run time and memory:

```bash
sres
bash slurm/submit_graph_grid.sh \
  --gpu GPU_FROM_SRES \
  --time MEASURED_TIME \
  --mem MEASURED_MEMORY \
  --max-parallel 2
```

Use `--dry-run` with the same required options to print the two `sbatch`
commands without submitting. The script refuses to reuse an existing named
output directory.
