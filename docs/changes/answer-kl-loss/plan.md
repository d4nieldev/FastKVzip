# Add a KL loss option to answer training

## Why

Right now answer training has one loss: NLL on the answer tokens, using the compressed cache.

The answer tokens themselves come from the frozen LLM run greedily with the **full, uncompressed** cache. So
NLL only ever sees the single token that run picked. Its full probability distribution gets thrown away,
even though it says a lot more about what compression actually broke.

This adds a second objective you can switch to: **KL between the full-cache distribution and the
pruned-cache distribution**, on the answer tokens. Instead of "predict the token the full-cache run picked",
it becomes "match the whole distribution the full-cache run had".

There is no KL or distillation code anywhere in the repo today, so this is all new.

## Before anything else: the code has moved

Your local `main` is **34 commits behind** `origin/main`, and those commits rewrote the exact files this
touches — `train_graph_answer.py` (236 lines), `answer_training.py` (109 lines), plus a new 480-line
`test_answer_batching.py`. I re-checked every assumption against `origin/main`. Four things changed that
affect this work:

- **Training now batches questions.** `--gradient-accumulation-steps N` runs N questions per optimizer
  update. Each question backwards its own loss; a new `finish_answer_batch` averages the gradients and does
  one step. So "one question per step" is only true at the default N=1.
- **`--max-contexts` was removed.**
- **The saved W&B step counter is gone** — logging now uses the standard W&B Step axis.
- **There's an established pattern for exactly your checkpoint request.** A helper called
  `normalized_answer_resume_config` fills in defaults for fields that old checkpoints don't have. Two
  recently added flags already use it. The "missing loss field means nll" rule drops straight into it.

I'll branch from `origin/main`, so none of this is a problem — but it means **your local `main` and its
uncommitted edits stay behind.** See the setup note at the bottom.

## What you'll get

A new flag, sitting next to `--gradient-accumulation-steps`:

```
--loss nll    # default, exactly what happens today
--loss kl     # train on KL instead
```

The two are **mutually exclusive** — never added together. With `--loss nll` nothing changes at all.

## How it works

**The extra forward pass.** In KL mode, each question runs the LLM twice instead of once: once with the full
uncompressed cache (no gradients), once with the pruned cache (as today). KL is computed between the two
distributions on the answer positions only — the same positions NLL already uses.

Direction is forward KL, full ‖ pruned, temperature 1. That penalizes the pruned cache for missing anything
the full cache thought was plausible.

Conveniently, the full cache is still sitting there untouched when the pruned forward happens — the pruned
cache is built as a separate copy — so the second pass needs no extra prefill. It's told to leave the cache
as it found it.

**Naming.** The repo already uses "teacher" to mean the frozen LLM wrapper. I'll call the KL target
*full-cache logits*, not *teacher logits*, so the two don't get confused in review.

**How it's averaged.** KL is averaged over a question's answer tokens, then averaged equally across
questions in a batch — identical to how NLL works today, so the two losses stay directly comparable and
gradient accumulation keeps behaving the same way.

**Numerics.** KL is a difference between two log-probabilities that are each around 10, where the signal we
care about is around 0.001. bfloat16 would wipe that out, so KL is computed in float32. NLL stays in
bfloat16 so `train/answer_nll` remains comparable between nll and kl runs.

**Cost.** The full-cache pass attends over the whole context instead of the pruned 10-30%, so roughly 1.7×
a pruned forward. Steps are dominated by the scorer, so expect under 20% slower overall.

**Memory is the thing to watch.** The experiments skill pins the recommended settings at 86 of 96 GiB — 89%
— and warns a long context could OOM. KL adds roughly 2.4 MB per answer token, so ~1.2 GB for a 512-token
answer, plus the extra pass's own transients. **Run a minimal pilot and check `nvidia-smi` before
submitting any grid.**

## What gets logged and how checkpoints are picked

In KL mode you'll see `train/answer_kl` per step and `validation/answer_kl` at each eval, alongside the NLL
and accuracy you already get. I'm logging the training KL per step because otherwise the thing you're
actually minimizing would be invisible during a run — you'd only see NLL moving.

`best.pt` follows **validation KL** when training with KL, and validation NLL when training with NLL. You
select on what you optimize. Validation KL is token-weighted, exactly like validation NLL.

There's one more place the validation metric is consumed. If you ever run with
`--gate-lr-scheduler ReduceLROnPlateau` (PyTorch's stock "cut the LR when the metric stops improving"
scheduler), the trainer feeds it the validation metric; that gets the same one-line switch. Note this is
currently dead code for you — LR schedulers are off by default and every documented run uses either
`LinearWarmupCosineLR` or nothing.

One operational note: **use a separate `--output-dir` for nll and kl runs.** The script reuses a saved W&B
run ID per output directory, so pointing both at the same folder would merge two different objectives into
one W&B run and overwrite `best.pt` with a best chosen under the other loss.

## Checkpoints and resume

Per your comment: the checkpoint always records which loss it was trained with. A checkpoint that predates
this feature and has no loss field is treated as `nll` — using the existing `normalized_answer_resume_config`
helper that already does this for two other fields.

That means:

- **Old checkpoints resume normally.**
- **Resuming a KL checkpoint doesn't need `--loss`** — it picks it up from the checkpoint.
- **You can't switch loss mid-resume.** Trying to resume an NLL checkpoint with `--loss kl` fails with a
  clear message, and fails *early* — before W&B starts and before the 7B model loads.
- **Warm-starting** via `--graph-checkpoint` ignores the saved loss and uses the command line.

The internal objective version string stays as it is. Bumping it would make every existing answer checkpoint
unresumable, which is worse than the string being slightly inaccurate — the new loss field is what tells the
two objectives apart.

## Files

| File | What changes |
|---|---|
| `prefill/graph/answer_training.py` | new KL function, next to the existing NLL one (which is left alone) |
| `prefill/graph/__init__.py` | export it |
| `prefill/train_graph_answer.py` | the `--loss` flag, the second forward pass, logging, checkpoint field, best-model selection, batch averaging |
| `prefill/tests/test_answer_training.py` | tests for the KL math |
| `prefill/tests/test_graph_answer_train_cli.py` | flag, checkpoints, resume, logging allowlist, two-forward step |
| `prefill/tests/test_answer_batching.py` | KL under gradient accumulation |
| `prefill/README.md` | document the flag in the stage-2 section |
| `docs/changes/<slug>/plan.md` + `decisions.md` | the repo's per-change record convention |

Nothing changes in `slurm/` — the submitter forwards unknown flags through. `docs/experiment-results.md` is
left alone; its stage-2 section is accurate history for completed NLL runs.

## How I'll verify it

**Unit tests on the KL math.** The important ones:

- The value matches a hand-computed number, using deliberately lopsided distributions so that
  **swapping the two arguments gives a different answer**. Getting the direction backwards is the single
  most common bug in this kind of code, and a symmetric test case would pass either way.
- Only answer positions contribute; everything before and after gets exactly zero gradient.
- Zero for identical distributions, never negative.
- **The full-cache tensors don't break autograd.** This is the one I care most about: PyTorch's KL function
  holds onto the target tensor for the backward pass, and tensors produced in inference mode can't be held
  that way. It would crash on the very first step on the cluster, and none of the existing CPU-based tests
  would catch it.

**Tests on the training step.** That KL mode runs exactly two model forwards per question and NLL mode runs
exactly one; that one pass sees the uncompressed cache and the other the compressed one; that both are told
the same token positions; that the full cache is left untouched afterwards; and that the KL gradient
actually reaches the scorer rather than dying at the pruning step.

**A test under gradient accumulation** — that a KL batch equals the mean of its questions' KLs, mirroring
the existing test that does this for NLL with unequal answer lengths.

**Tests on config and resume.** Each of the four resume cases above, plus that `best.pt` is selected by KL
in KL mode — using a case where KL improves while NLL gets worse, which proves selection genuinely switched
rather than coincidentally agreeing.

**Existing tests stay green.** A few need updating because they assert exact things that now have a second
variant — the hardcoded logging allowlist, and a fake model that asserts it's called exactly once with an
exact argument list. Those same tests are the reason I'm confident NLL mode stays unchanged.

Run with `cd prefill && python -m pytest tests/ -q`.

**Then a real pilot on the cluster**, because the memory and inference-mode issues can only show up there: a
minimal run with `--loss kl`, confirm it doesn't crash, KL is finite and non-negative, NLL is still logged,
and check peak GPU memory. Then a short run with validation enabled to confirm `validation/answer_kl`
appears, `best.pt` gets written, and resuming without `--loss` picks KL back up.

## Setup

1. Branch from `origin/main` in a dedicated worktree (`.claude/worktrees/`), on a branch named for this
   feature. That gets the latest code without touching your main checkout.
2. Implement, test, commit the code plus `docs/changes/<slug>/plan.md` and `decisions.md`.
3. Push and open a PR.

**Heads up on your working tree:** you have uncommitted edits to `prefill/README.md`, `prefill/data/load.py`
and `docs/experiment-results.md`, plus untracked files including `AGENTS.md`. `origin/main` has since
rewritten `prefill/README.md` heavily, so a plain `git pull` on your main checkout would likely conflict.
Working in a worktree off `origin/main` sidesteps that entirely — but **those local edits won't be in this
PR**, and you'll still need to deal with them separately. `AGENTS.md` in particular doesn't exist on
`origin/main`; the equivalent guidance now lives in `.agents/skills/running-fastkvzip-experiments/`, so I'll
put the memory caveat there instead if it belongs anywhere.
