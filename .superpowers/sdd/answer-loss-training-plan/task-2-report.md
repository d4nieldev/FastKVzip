# Task 2 report — answer-training primitives

## Status

Implemented the focused reusable primitives in `prefill/graph/answer_training.py`
with public exports from `prefill/graph/__init__.py`. No trainer, optimizer step,
dataset registry, or speculative framework was added.

## TDD evidence

Baseline before Task 2 edits:

```text
$ cd prefill
$ /private/tmp/fastkvzip-answer-py312/bin/python -m pytest -q tests
147 passed in 4.58s
```

RED after adding `prefill/tests/test_answer_training.py` and before creating the
production module:

```text
$ cd prefill
$ /private/tmp/fastkvzip-answer-py312/bin/python -m pytest -q tests/test_answer_training.py
16 failed in 0.86s
```

Every failure reported the expected missing `graph.answer_training` module.

The first GREEN attempt reached 16 passing tests and one invalid test fixture:
the fixture gave keys and values different head dimensions. After correcting
that fixture, the exact-hard-forward assertion exposed a real implementation
issue. The left-associated floating expression `1 + p - p.detach()` produced
the value immediately below 1 for the selected probability used by the test:

```text
left-associated: 0x1.fffffffffffffp-1, equal to 1: False
grouped STE:     0x1.0000000000000p+0, equal to 1: True
```

The implementation now evaluates the binding multiplier as
`ones + (p - p.detach())`. It is algebraically the required
`1 + p_selected - p_selected.detach()`, preserves the same gradient, and makes
the hard forward bitwise equal to ordinary gathered values.

Focused GREEN:

```text
$ cd prefill
$ /private/tmp/fastkvzip-answer-py312/bin/python -m pytest -q tests/test_answer_training.py
17 passed in 1.38s
```

Full-suite GREEN before final verification:

```text
$ cd prefill
$ /private/tmp/fastkvzip-answer-py312/bin/python -m pytest -q tests
164 passed in 4.82s
```

## Design notes

- `retention_ratio` uses an explicitly supplied `random.Random` for resumable
  uniform sampling. Global-linear interpolation is a pure function of the
  clamped flattened global step and fixed total horizon, decaying max to min.
- `fallback_validation_split` order-deduplicates once and reserves the last
  `ceil(10%)`, rejecting fewer than two unique requested examples.
- `retained_token_count` is exactly `max(1, floor(ratio * N))` for `N > 0`.
- `score_context_subgraphs` invokes the existing streamed
  `ImplicitGraphScorer` independently for each local subgraph and concatenates
  its existing `[layers, 1, KV heads, tokens]` raw-score shape.
- `global_topk_indices` performs one vectorized raw-score `torch.topk` over the
  complete concatenated context, independently per layer/head, then restores
  chronological cache order. It has no softmax, protected window, or local
  selection.
- `compact_context_kv` receives an explicit context range. Tokens outside that
  range are copied unchanged and are outside the context budget. Only selected
  context values receive the mass-k STE; selected keys are plain gathers.
  Inference tensors captured by the existing prefill path are cloned into
  normal tensors so the answer forward can remain differentiable.
- `answer_objective` applies the causal shift and computes mean CE, summed NLL,
  correct-token count, token count, and accuracy over answer target tokens only.
- `replay_score_gradients` recomputes the differentiable chunked scorer and
  applies the detached external VJP. It neither clears gradients nor steps an
  optimizer, allowing the future trainer to make exactly one update per example.
- `freeze_llm` disables LLM parameter gradients and switches it to evaluation
  mode without wrapping the answer forward in `no_grad`/`inference_mode`.
- `validate_answer_training_model_identity` accepts Qwen and Llama model IDs or
  runtime identities and gives Gemma3 a specific early rejection.

## Test coverage

`prefill/tests/test_answer_training.py` covers:

- uniform range and explicit RNG resume determinism;
- max-to-min global-linear endpoints, monotonicity, and clamping;
- floor rounding and the one-token minimum;
- ordered deduplication and fallback holdout;
- independent subgraph scoring and one global per-head top-k with no protected
  positions;
- physical context compaction while preserving prefix/suffix tokens;
- bitwise hard-forward K/V equivalence, value-only STE, and nonzero retained and
  evicted score gradients;
- validation hard compaction without a softmax graph;
- normal differentiable copies of inference-created cache tensors;
- answer-only causal NLL and argmax token accuracy with sums/counts;
- external replay gradient equality against direct autograd on a tiny real
  `ImplicitGraphScorer`;
- frozen LLM parameters, value gradients, and scorer-only updates;
- Qwen/Llama acceptance and explicit Gemma3 rejection.

## Files

- `prefill/graph/answer_training.py` — focused primitives.
- `prefill/graph/__init__.py` — public exports.
- `prefill/tests/test_answer_training.py` — focused behavior tests.
- `.superpowers/sdd/answer-loss-training-plan/task-2-report.md` — this report.

## Concerns

No Task 2 blocker remains. Task 3 must update runtime cache metadata (such as
seen-token length) after installing the returned physical K/V tensors and must
call model-identity validation before constructing the model. Those orchestration
responsibilities intentionally remain outside this primitive module.
