# Task 4 report — answer-training Slurm entry point and documentation

## Design

- `submit_train_graph_answer.sh` submits exactly one job. It requires explicit
  GPU, time, memory, and scratch values, parses split and `--flag=value` forms,
  forwards trainer arguments in a Bash array, and prints a safely quoted array
  command for `--dry-run` before it creates logs or invokes `sbatch`.
- The helper owns `--output-dir`; `OUTPUT_ROOT/RUN_NAME` is passed to the batch
  script, with durable project `graph_checkpoints/answer` as the default.
  `--answer-cache-dir` is otherwise left unchanged and never derives from
  scratch.
- The batch script uses `SLURM_TMPDIR` only for Hugging Face, datasets, and
  Transformers caches. Checkpoints and W&B remain under the durable run
  directory. It deliberately does not require `MODEL_ID`, so resume and graph
  checkpoint modes determine the model normally.
- The README and experiment guide cover local random/graph-checkpoint/resume
  paths, lazy cache behavior, resource selection after `sres`, durable output
  selection, and the matching Agentic pair-head, zero-window evaluation.

## TDD evidence

RED, before either Slurm entry point existed:

```text
$ /private/tmp/fastkvzip-answer-py312/bin/python -m pytest -q prefill/tests/test_answer_training_slurm.py
7 failed
```

The failures were the expected missing helper/batch files. The focused tests
exercise the real helper with a fake venv and a PATH without `sbatch`; they
assert dry-run resource/job/batch/forwarded durable arguments, all four missing
resource failures (including `--tmp`), and helper-owned output rejection.

GREEN:

```text
$ /private/tmp/fastkvzip-answer-py312/bin/python -m pytest -q prefill/tests/test_answer_training_slurm.py
7 passed in 0.15s
```

## Verification

```text
$ cd prefill && /private/tmp/fastkvzip-answer-py312/bin/python -m pytest -q tests
187 passed in 4.96s

$ bash -n slurm/submit_train_graph_answer.sh slurm/train_graph_answer.sbatch
exit 0

$ git diff --check
exit 0
```

No Slurm job was submitted.

## Files

- `slurm/submit_train_graph_answer.sh`
- `slurm/train_graph_answer.sbatch`
- `prefill/tests/test_answer_training_slurm.py`
- `prefill/README.md`
- `docs/graph-fastkvzip-experiments.md`
- `.superpowers/sdd/answer-loss-training-plan/task-4-report.md`

## Review corrections

The Task 4 review found two issues: the new local and Slurm examples omitted
the trainer's required `--validation-retention-ratio`, and split resource flags
accepted a following option as their value.

### RED/GREEN

RED added an eight-case matrix for every resource's split and equals forms and
a twelve-case matrix for trailing, next-option, and empty-equals values:

```text
$ /private/tmp/fastkvzip-answer-py312/bin/python -m pytest -q prefill/tests/test_answer_training_slurm.py
12 failed, 11 passed in 1.02s
```

The failing malformed cases showed the helper submitting dry-run commands for
trailing/next-option values and returning only generic usage for empty equals
values. The helper now rejects empty values and values beginning with `--`
before assigning resources. Each new random, graph-checkpoint, and resume
example supplies `--validation-retention-ratio 0.2`; the docs also state that a
resume must use its checkpoint-compatible saved value.

GREEN:

```text
$ /private/tmp/fastkvzip-answer-py312/bin/python -m pytest -q prefill/tests/test_answer_training_slurm.py
23 passed in 0.62s

$ cd prefill && /private/tmp/fastkvzip-answer-py312/bin/python -m pytest -q tests
203 passed in 5.80s

$ bash -n slurm/submit_train_graph_answer.sh slurm/train_graph_answer.sbatch
exit 0

$ git diff --check
exit 0
```
