# Answer KL loss decisions

## D1 — Ordering, not a guard, keeps the reference out of inference mode

- **Plan gap or deviation:** The plan named the hazard and required a test for it
  (`F.kl_div` holds its target for the backward pass, and an inference-mode tensor cannot
  be held that way), but left the remedy open.
- **Decision and effect:** Add no guard. `answer_kl_objective`
  ([`prefill/graph/answer_training.py`](../../../prefill/graph/answer_training.py)) runs
  `F.log_softmax` on the reference before `F.kl_div`, so what `kl_div` holds is the
  softmax output rather than the reference itself.
- **Reason and tradeoff:**
  - Running `log_softmax` outside inference mode on an inference tensor already yields an
    ordinary tensor. This is the same documented invariant `compact_context_kv` relies on.
  - The obvious alternative, `_normal_tensor`, is both redundant and costly here:
    validation runs entirely under inference mode, so it would clone a
    `[1, answer, vocab]` tensor for every validation question.
  - The ordering is load-bearing and invisible, so a code comment records that the
    reference must never reach `kl_div` unsoftmaxed.
- **Coverage:** `test_answer_kl_accepts_inference_mode_reference_and_stays_differentiable`
  builds the reference inside `torch.inference_mode()` and backwards through it. Adding
  `_normal_tensor` changes no test outcome, which is the evidence it is unnecessary.
- **Status:** Agent decision.

## D2 — The reference forward rolls the cache back instead of copying it

- **Plan gap or deviation:** The plan fixed *that* a full-cache forward happens but left
  *how* it avoids disturbing the pruned forward open.
- **Decision and effect:** `full_cache_logits`
  ([`prefill/train_graph_answer.py`](../../../prefill/train_graph_answer.py)) calls the
  wrapper with `update_cache=False`, so the wrapper's existing `kv.slice` restores the
  prefilled length. It runs before compaction, and reuses one `_answer_cache_position`
  helper shared with the pruned forward.
- **Reason and tradeoff:**
  - Copying the cache would duplicate every layer's keys and values, which is the single
    largest allocation in the step.
  - The rollback path is already the established pattern for scoring against a full cache.
  - Sharing the position helper is what makes the two distributions comparable; computing
    it twice would let them drift apart silently.
- **Coverage:** `test_kl_step_reads_the_full_cache_once_and_restores_it` asserts the
  reference sees the uncompacted cache, the pruned forward sees the compacted one, both
  receive equal positions, and the cache is restored afterwards.
- **Status:** Agent decision.

## D3 — NLL stays reported in KL mode, outside the autograd graph

- **Plan gap or deviation:** The plan promised NLL remains logged in KL mode but did not
  say how, given the frozen W&B allowlist and the cost of an unused graph.
- **Decision and effect:** `train_answer_example` still calls `answer_objective` in KL
  mode, wrapped in `torch.no_grad()`.
- **Reason and tradeoff:** Keeps `train/answer_nll` and `train/answer_token_accuracy`
  comparable across objectives at negligible cost, without retaining a
  `[1, answer, vocab]` cross-entropy graph that is never backwarded.
- **Status:** Agent decision filling a plan gap.

## D4 — Twin metric allowlists rather than extended ones

- **Plan gap or deviation:** The plan promised the two KL metrics but left the allowlist
  design open, expecting "the hardcoded logging allowlist" test to need updating.
- **Decision and effect:** Add `TRAIN_KL_LOG_KEYS` and `VALIDATION_KL_LOG_KEYS` as supersets
  and select between them on `options.loss`. The existing frozensets are unchanged.
- **Reason and tradeoff:** An NLL run must keep emitting exactly the key set it emits today,
  so its W&B history stays comparable with completed grids. Extending the shared sets in
  place would have made NLL runs fail their own allowlist assertion.
- **Status:** Agent decision; deviation from the planned edit.

## D5 — Selection takes an explicit loss, and the cursor key is reused

- **Plan gap or deviation:** The plan left open whether to rename
  `cursor["best_validation_nll"]`, and did not specify how selection learns which metric to
  use.
- **Decision and effect:** Keep the key. `update_validation_cursor` and
  `validation_log_metrics` take an explicit `loss` keyword defaulting to `"nll"`; a new
  `validation_selection_metric` resolves the number and raises if a KL run has no
  divergence.
- **Reason and tradeoff:**
  - `restore_training_state` deep-copies the saved cursor without merging defaults, so a
    rename would break every existing checkpoint.
  - The name is accurate per run because the resolved loss cannot change across a resume
    chain: `normalized_answer_resume_config` guarantees the saved config always carries a
    `loss`, so a strict resume always inherits it and a conflicting `--loss` is rejected.
    A code comment records that reasoning.
  - An explicit keyword rather than sniffing for a non-`None` `answer_kl` keeps the
    existing validation test, which passes a namespace with no such attribute, working.
- **Status:** Agent decision.

## D6 — The KL path is additive, so no existing test needed changing

- **Plan gap or deviation:** The plan expected to update the hardcoded metric allowlist test
  and the fake model that asserts it is called exactly once with an exact argument list.
- **Decision and effect:** Neither was touched. `answer_kl` and `answer_kl_sum` are
  defaulted `None` fields appended after `prefix_ids`, every new keyword defaults to the NLL
  behavior, and the reference forward is conditional.
- **Reason and tradeoff:** Those two tests are the strongest evidence that NLL mode is
  unchanged; rewriting them would have destroyed the evidence they provide. A new
  `test_nll_step_never_reads_the_full_cache` pins the single-forward behavior directly.
- **Status:** Agent decision; a deliberate non-change that explains why the diff touches no
  existing test.

## Validation results

From the worktree's `prefill/` directory, on a CPU-only Python 3.11 environment built for
this change (PyTorch 2.14.0, Transformers 4.57.6, W&B 0.30.0; the repository pins no
versions and no prepared environment existed locally):

```bash
python -m pytest tests/ -q
```

Result: **691 passed**, including 17 new tests. The 674 pre-existing tests all pass
unchanged. The new coverage includes the KL value against a hand-computed divergence with
asymmetric distributions, the argument-order asymmetry, the `(pruned - full) / tokens`
gradient with exactly zero gradient outside the answer slice, float32 promotion from
bfloat16, inference-mode references, both forward counts, equal-question batch averaging,
token-weighted validation aggregation, selection that follows KL where it disagrees with
NLL, and all four resume and warm-start paths.

The KL math tests were mutation-checked: flipping the `kl_div` argument order and removing
the float32 promotion each fail them. Removing `_normal_tensor` did not, which is what
established D1.

No GPU or cluster validation was performed. Peak memory and step time under `--loss kl`
are unmeasured and need a single-context pilot before any grid submission.
