# Running implicit graph experiments

This guide uses the implicit whole-context mixer in this PR.
Run commands from the repository root on the BGU cluster.

## One-time setup

```bash
cd /home/danieloh/FastKVzip-implicit
git switch feature/whole-context-graph
git pull --ff-only
mkdir -p .slurm/logs
export FASTKVZIP_VENV=/home/danieloh/.venvs/fastkvzip
export MODEL_ID=Qwen/Qwen3-8B
```

The batch scripts activate this existing environment. They do not rebuild
FlashAttention. W&B is online by default. Do not put its API key in a script.

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
slice or one raw optimizer call. By default, `last.pt` is saved every training
context and validation runs every epoch. A validation sweep never writes
`last.pt` after each held-out context. `best.pt` is written when its mean BCE
improves. Use `last.pt` for a pilot or resume. Prefer `best.pt` for final
evaluation.

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
| validation cadence | `--eval-strategy`, `--eval-every` |
| memory/speed knobs | `--graph-microbatch-size`, `--token-microbatch-size` |

Joint mode is the default. Two-phase mode updates the gate in token slices,
then updates the mixer once per context. Scheduler arguments must be JSON.
`--max-contexts` also counts training contexts only.

Use `steps` for a number of completed training contexts. Use `epochs` for
completed training epochs. The defaults are `--save-strategy steps
--save-every 1` and `--eval-strategy epochs --eval-every 1`.

```bash
--gate-lr-scheduler StepLR --gate-lr-scheduler-kwargs '{"step_size": 1, "gamma": 0.5}'
```

Do not reuse a run name. Do not let two jobs fill the same missing shared
teacher cache. The default per-job scratch cache avoids that race.

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

```bash
sres
sbatch --gpus=rtx_pro_6000:1 slurm/eval_graph.sbatch gd32-seed0-squad \
  --graph-checkpoint runs/graph/gd32-seed0/best.pt \
  --data squad \
  --idx 0 \
  --num 1
```

The script gives the result a unique graph tag. Results are under
`prefill/results/<data>/`. The checkpoint restores the model, prefix, and
prefill settings. Do not add architecture flags to evaluation.

Training W&B logs `validation/bce` once per validation sweep. It is the mean
across the four held-out contexts. It does not log task accuracy or generation
metrics; standalone evaluation writes those benchmark results.

## Monitor a job

Save the job ID printed by `sbatch`.

```bash
squeue --jobs=JOB_ID -o '%.18i %.9P %.20j %.2t %.10M %.6D %R'
scontrol show job JOB_ID
tail -f .slurm/logs/graph-train-JOB_ID.out
sacct --jobs=JOB_ID --format=JobID,State,ExitCode,Elapsed,AllocTRES,MaxRSS
```

Use `sstat` only while the job is running. Do not resubmit a pending job.
Read its reason first.

## Before a real grid

Run one pilot per new resource shape. Record its W&B link, checkpoint path,
commit, and `sacct` result. Then choose the full time and memory request.

Submit one job per configuration for now. Add a throttled Slurm array only
after the exact grid rows and resource request are fixed.
