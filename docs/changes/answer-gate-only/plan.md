# Gate-only answer training: fine-tune the FastKVzip gate

## Context

Answer training normally starts from a **stage-1 graph checkpoint** — gate and
mixer together — passed with `--graph-checkpoint`. Random initialization is only
the fallback when no checkpoint is given. (My earlier sentence said it always
starts random; that was wrong.)

What is missing is a way to start from the **FastKVzip gate alone**. Today the
gate can only arrive inside a full graph checkpoint, so there is no way to ask:
*does the graph mixer earn its keep, or would answer-supervising the plain
FastKVzip gate get most of the way there?*

This adds that third starting point, and a mode where no mixer exists at all.

## What you'll get

Two independent flags on answer training:

- **`--gate-checkpoint fastkvzip`** — start from the released FastKVzip gate
  (downloaded from HF, same as stage 1 already does). A local file works too.
- **`--no-graph-mixer`** — no mixer is built. Only the gate is scored and only
  the gate trains.

Your run is both:

```bash
python -B train_graph_answer.py --model Qwen/Qwen2.5-7B-Instruct-1M \
  --gate-checkpoint fastkvzip --no-graph-mixer \
  --validation-retention-ratio 0.2 --output-dir .../answer/fastkvzip-gate-only
```

Everything else behaves exactly as today — same data, same losses (`nll`/`kl`),
same retention schedule, same accumulation, same resume, same W&B. Passing
neither flag changes nothing.

**Out:** the usual `best.pt` / `last.pt`, which resume normally. Because the
result *is* a FastKVzip gate, you evaluate it on the upstream path:

```bash
python -B eval_chunk.py -g .../answer/fastkvzip-gate-only/best.pt \
  -d scbench_qa_eng -r 0.3 --tag gate-only
```

## Answering your question: can `eval_chunk.py` take a local path?

**No, not today.** `-g` is treated as a *name inside the HF repo*
`Jang-Hyun/Fast-KVzip`; it gets turned into `<model>/<name>.pt` and downloaded.
There is a local fallback in the code but it is broken (an unexpanded `~`, and it
drops the `.pt`). The gate's sink size is also read from the *filename*, not from
the weights. So `-g` learns to accept a path — that part of the work is
unavoidable and is included.

Three traps come with it, all of which would have produced quietly wrong numbers:

- **Path text is currently pattern-matched.** If the `-g` value contains `snap`
  or `expect` anywhere, the evaluator silently switches the eviction structure.
  Any checkpoint sitting under a `snapshots/` directory — which is how the HF
  cache is laid out on the cluster — would have been evaluated with different
  eviction than its baseline. The path branch will not be pattern-matched.
- **Result identity.** Results are named from the last path segment, so every
  run evaluated by path would be tagged `best.pt` and collide in the results
  directory. The run directory name goes into the tag instead.
- **Layer order.** A checkpoint stores gate weights keyed `"0."…"35."`. Sorted as
  text that is 0, 1, 10, 11, 2… — every layer's gate silently assigned to the
  wrong layer, with no shape error to catch it. Loading sorts numerically and
  asserts the full layer range, with a test.

## The decision that affects your numbers: precision

This is the one thing worth your attention.

Today the gate is scored in **fp32** during training — not because anyone chose
that, but because the mixer's output is fp32 and promotes the gate's input. The
upstream evaluator scores the gate in **bf16**. I had that gap measured on real
shapes: **~1.6e-3 relative score drift, and ~1% of the retained set changes** at
10% retention.

Two consequences:

1. **The ablation has to hold precision fixed.** If removing the mixer also drops
   training to bf16, the gate-only arm differs from the mixer arm by precision as
   well as by architecture, and the comparison is confounded by roughly the same
   magnitude as the effect you are looking for. Gate-only training stays in fp32.
2. **Eval has to match training.** A fine-tuned gate must reproduce its own
   validation scores. The evaluator will score a gate in the precision that gate
   was saved in — a no-op for the released bf16 gate, fp32 for a fine-tuned one.

Without both, any "fine-tuned vs. base" difference smaller than ~1% retained-set
churn is noise.

## Other behavior worth agreeing on

- **`--no-graph-mixer` rejects the mixer's knobs.** `--graph-dim`, `--alpha-init`,
  `--gram-normalization`, `--leaky-relu-slope`, `--mixer-lr` and the mixer LR
  scheduler all fail fast rather than sit meaninglessly in the checkpoint and the
  W&B config. The checkpoint records "no mixer" as a single fact, so a resume
  infers the mode and you never have to re-type the flag.
- **`--subgraph-size` stays and still matters.** With no mixer the scores no
  longer depend on how the context is partitioned — but the flag is still the
  memory control, because gradients are replayed and freed per subgraph. Keep
  using your tuned values. Your `AGENTS.md` scorer-width table does not transfer
  to this mode; the pilot re-measures it.
- **`--gate-checkpoint` is exclusive with `--resume` and `--graph-checkpoint`** —
  they are three alternative ways to start, and combining them is rejected up
  front.
- **Warm-starting from a previous *training* checkpoint runs in fp32**, since
  that is the precision its weights were saved in. Fine, but it is slower and
  heavier than `fastkvzip`, and it is locked in for the run.
- **W&B keeps the same metric schema.** The three mixer metrics
  (`mixer_learning_rate`, `mixer_grad_norm`, `mean_alpha`) report `0.0` rather
  than disappearing, so gate-only and mixer runs chart against each other. They
  are true values, not placeholders: gate-only *is* the zero-alpha limit.
- **`eval_graph.py` / `eval_graph_chunked.py` refuse a gate-only checkpoint**
  with a message pointing at `eval_chunk.py -g` (your call). Worth knowing
  operationally: the experiments runbook pairs an `afterok` graph-eval job with
  every training job, so for these runs you queue an `eval_chunk` job instead.

## Where the work lands

The scorer learns to exist without a mixer; answer training gets the two flags
and the rules above; checkpoint validation accepts a mixer-free checkpoint on
resume without loosening anything for normal ones; and the upstream gate loader
learns paths, layer ordering, sink-from-weights, and precision.

`prefill/graph/model.py`, `prefill/train_graph_answer.py`,
`prefill/attention/gate.py`, `prefill/graph/evaluation.py`,
`prefill/graph/training.py`, `prefill/args.py`, plus `prefill/README.md` and a
`docs/changes/answer-gate-only/` record. Nothing in `slurm/` changes — the
submitter forwards new flags as-is.

## How it gets verified

Tests (`cd prefill && python -m pytest tests/ -q`, CPU-only), covering behavior:

- Gate-only scoring equals the plain FastKVzip gate on the same hidden states,
  and equals a mixer scorer with `alpha` zeroed — the exact-limit check.
- `--subgraph-size` does not change gate-only scores.
- A gate-only run trains, saves, and resumes; a mixer run is bit-for-bit
  unaffected.
- A gate written by training reloads through `eval_chunk.py`'s loader and
  reproduces the training-time scores — layer order, sink, and precision all
  exercised. (`attention/gate.py` has no tests at all today.)
- The rejected flag combinations fail with clear messages.

Then on the cluster:

1. A 3-context pilot to confirm the real path runs and to re-measure memory and
   step time, which should drop noticeably without the mixer's Gram work.
2. `eval_chunk.py` on the fine-tuned gate vs. `-g fastkvzip`, same task and
   ratio — the number you actually want.
3. A zero-step checkpoint evaluated against `-g fastkvzip`, which must match:
   proof that the save/load/precision round trip is faithful before reading
   anything into the fine-tuned result.

## Setup

Worktree and branch `feature/answer-gate-only` off `origin/main` (local `main` is
in sync). The approved plan and the implementation decisions are committed with
the code under `docs/changes/answer-gate-only/`.
