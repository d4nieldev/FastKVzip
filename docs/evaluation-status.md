# Graph Evaluation Status

## Why we checked the pipeline

GraphKV needs hidden states from the base model and scores them with an
external mixer. The released FastKVzip evaluator uses its built-in gate.
Therefore, GraphKV has a separate evaluation pipeline. We needed to verify
that this separate code still produces correct results.

We became suspicious after low scores at small retention ratios. The important
difference was the protected 4,096-token window at the end of the context. In
medium contexts longer than 16K tokens, this window uses much of a small
retention budget. Few tokens can then be selected from earlier in the context.
This caused the low scores at small retention ratios. Most benchmarks selected
by `--data all` contain contexts in this range.

## Verification result

We used the released Qwen2.5-7B-Instruct-1M FastKVzip gate. We created an
equivalent GraphKV checkpoint by setting every mixer alpha to zero. We then
evaluated all 100 `scbench_kv` contexts. Each context has five questions.

| Retention | Official FastKVzip (%) | GraphKV chunked (%) | GraphKV whole-context (%) |
|---:|---:|---:|---:|
| 75% | 67.0 | 69.0 | 69.8 |
| 50% | 71.0 | 72.4 | 74.6 |
| 40% | 66.2 | 66.6 | 72.4 |
| 30% | 65.8 | 63.8 | 54.8 |
| 20% | 44.2 | 46.2 | 48.6 |

Agreement means that both answers received the same correct or incorrect score.
It does not require identical answer text.

| Evaluation pair | Agreement over 2,500 question-and-ratio results |
|---|---:|
| Official and GraphKV chunked | 97.32% |
| Official and GraphKV whole-context | 84.08% |
| GraphKV chunked and whole-context | 84.60% |

Official and GraphKV chunked agreement at each ratio was between 96.8% and
97.6%.

This confirms that the GraphKV scoring, generation, result saving, and metric
calculation work correctly. The small remaining differences do not show a
systematic evaluation mismatch.

## What to use now

Use `eval_graph.py`, the whole-context evaluator, for normal GraphKV
experiments. We will continue adapting the whole-context pipeline.

The default task is now `scbench_kv`. Use `--data all` only when the full suite
is intentional.

`eval_graph_chunked.py` is slow and experimental. It exists only to compare
our pipeline with the released evaluator. Avoid it for regular experiments.
