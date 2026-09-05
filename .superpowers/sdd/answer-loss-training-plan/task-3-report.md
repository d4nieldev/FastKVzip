# Task 3 report — answer-supervised training entry point

## Status

Implemented `prefill/train_graph_answer.py` as the answer-loss orchestration layer.
It reuses the existing graph scorer, Task 2 selection/objective/VJP primitives,
regular trainer initialization/config/cadence/W&B helpers, graph
optimizer/scheduler/checkpoint helpers, `DataWrapper`, and evaluation checkpoint
normalization.

## Design

- The CLI has mutually exclusive full resume and weights-only graph initialization.
  Model identity and checkpoint model matching are resolved before W&B, model
  construction, or dataset streaming. Fresh initialization requires a Qwen/Llama
  model; Gemma3 is rejected early.
- Resume inherits and verifies the complete answer-training configuration, loads
  scorer/optimizer/scheduler state with global RNG restoration, restores the data
  cursor, fixed retention horizon, separate uniform-retention RNG, exact prefix,
  best validation NLL, and W&B run ID. Graph initialization loads only scorer
  weights with `restore_rng=False`, retaining fresh optimizer/scheduler/cursor/RNG.
- The generic dataset loader receives split, runtime teacher, answer cache, start,
  and count. Native validation is used when declared. Otherwise the selected train
  range is loaded once, exact-deduplicated by context/question without copying
  context bodies, and its final `ceil(10%)` is reserved.
- Each example performs one full unpruned hidden/KV prefill, resolves any deferred
  teacher answer, scores independent subgraphs, performs one raw global per-head
  top-k, and physically compacts only the context. The query/template and teacher
  answer are then forwarded through the frozen LLM outside the context budget.
- The physical one-use cache keeps the original logical seen-token count and uses
  explicit original `cache_position` values while its K/V tensors have compacted
  physical length. This preserves RoPE positions. Keys are hard gathers; selected
  values use the Task 2 STE only during training.
- Answer-only backward terminates at a detached raw-score leaf, records score
  gradient health, replays one external VJP through the chunked scorer, then steps
  each gate/mixer optimizer and non-plateau scheduler exactly once. Validation is
  inference-only hard top-k, aggregates summed NLL/correct tokens by total answer
  tokens, and selects best checkpoints by validation NLL.
- Checkpoints use the existing top-level format unchanged and add objective,
  dataset/range/deduplication, answer generation/cache identity, STE, retention,
  horizon, validation ratio, and existing normalized graph fields to `config`.
  Dynamic progress and uniform RNG state live in `data_cursor`.
- W&B metric builders enforce the exact approved train and validation allowlists.
  Retention stays in config/checkpoints and is never logged as a metric.

## TDD evidence

Initial RED, before the production module existed:

```text
$ cd prefill
$ /private/tmp/fastkvzip-answer-py312/bin/python -m pytest -q tests/test_graph_answer_train_cli.py
13 failed in 3.06s
```

All failures were the expected missing `prefill/train_graph_answer.py` entry point.

Additional focused RED/GREEN cycles caught three orchestration defects:

- A weights-only graph checkpoint incorrectly inherited answer objective/data
  settings. The added test failed with `uniform != linear`; graph initialization
  now inherits only the binding model/architecture/dtype/subgraph/prefill defaults
  while starting fresh answer-training settings.
- Resume looked for `data` instead of the persisted `dataset` field. The new resume
  test failed with `agentic != unit-data`; saved data selection and incompatible
  override rejection now use the checkpoint field correctly.
- The first physical-cache implementation used compacted length as the logical
  seen length. The cache-position regression failed with a missing
  `cache_position`; the fixed path retains original seen length and forwards
  explicit original positions.

Focused GREEN:

```text
$ cd prefill
$ /private/tmp/fastkvzip-answer-py312/bin/python -m pytest -q tests/test_graph_answer_train_cli.py
15 passed in 3.73s

$ /private/tmp/fastkvzip-answer-py312/bin/python -m pytest -q \
    tests/test_answer_training.py tests/test_graph_training.py \
    tests/test_graph_eval.py tests/test_data_load.py
69 passed in 3.41s
```

## Verification

Fresh final verification from `prefill/`:

```text
$ /private/tmp/fastkvzip-answer-py312/bin/python -m pytest -q tests
179 passed in 4.77s

$ /private/tmp/fastkvzip-answer-py312/bin/python -m py_compile \
    train_graph_answer.py tests/test_graph_answer_train_cli.py \
    graph/answer_training.py train_graph.py graph/training.py \
    graph/evaluation.py eval_graph.py data/wrapper.py model/wrapper.py \
    attention/kvcache.py model/template.py
exit 0
```

`git diff --check` is run on the staged Task 3 files before commit.

## Files

- `prefill/train_graph_answer.py`
- `prefill/tests/test_graph_answer_train_cli.py`
- `.superpowers/sdd/answer-loss-training-plan/task-3-report.md`

## Concerns

No implementation blocker remains. The differentiable answer path is covered with
a tiny CPU scorer/cache/LLM double, including logical cache metadata and frozen-LLM
immutability; a real large-model GPU smoke run is intentionally outside Task 3 and
would require model weights, FlashAttention, and external dataset access.
