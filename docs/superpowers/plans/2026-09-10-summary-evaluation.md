# Summarization evaluation implementation record

Approved behavior: GovReport and PG-19 examples supply a complete context and the
same 400–500-word summary request to the existing evaluation commands. Support
whole-context graph, chunked graph, FastKVzip, and KVzip pruning. For each document,
generate N answers for each unique requested ratio plus ratio 1 if absent. Match
all N pruned answers one-to-one with all N full answers using ROUGE-L F1; report
ROUGE-1/2 on the same assignment. No extra samples or half-sample comparisons.

Only decoding controls: temperature, top-p, top-k, max-new-tokens. Greedy is
temperature=0 OR top-k=1, and greedy with N>1 must fail before model work. Prompt
prefill/pruning happen once per condition and generated cache state is restored
between samples. Persist each sample immediately; identify model/tokenizer,
rendered input/prefix, decoding settings, and pruning settings on resume.

## Tasks and ownership

1. Dataset ingestion and matched ROUGE metric — summary_data_metric agent.
2. Cache-preserving prompt sampler and decoding settings — cached_sampling agent.
3. Durable sample records and result aggregation — sample_results agent.
4. Shared evaluator integration and documentation — root.
5. Regression, integration, and independent review — root with reviewer agents.

## Integration decisions

- Worktree starts at updated origin/main `aded853`; original main checkout has
  user edits and was left unchanged after a blocked fast-forward pull.
- New tasks use shared sample storage alongside existing single-answer results;
  the existing results parser handles both formats.
- All summary dataset rows are loaded before formatted-prompt/model-limit
  filtering and selection, so --idx/--num apply to the eligible cohort.
- Default summary output limit is 1024. Existing non-summary defaults retain
  their benchmark-specific lengths unless --max-new-tokens is explicitly set.
- Multiple generations apply to summary datasets; existing single-answer tasks
  reject N>1 instead of silently discarding requested outputs.
- The summarization selector is opt-in; the existing `all` selector and approved
  107-benchmark scheduler grid retain their original inventory.
- Cache-provider callbacks retain the pruning implementations in their existing
  entry points; the shared evaluation/scoring layer does not know the pruner.

## Verification record

- Baseline: 92 existing graph, chunked, baseline, data, storage and parsing tests
  passed before shared evaluator integration.
- Isolated environment: /private/tmp/fastkvzip-summary-venv; tests run from prefill
  with PYTHONPATH=. to avoid the unrelated top-level utils.py module.
- Dataset/metric task: 48 tests passed.
- Full prefill suite: 490 tests and 12 subtests passed, including all four pruning
  paths, both chunked baseline variants, and existing single-answer evaluations.
- Actual Retain, Evict, and Hybrid cache classes were exercised with tiny CPU
  models; the CUDA append operation was replaced with its CPU equivalent.
- Independent review identified and verified fixes for full-bin coverage,
  excluded-source IDs, dynamic-cache memory retention, and exporting saved full
  pools into a shared reference cache on resume. Final review: no findings.
- Long-context CUDA execution has not been run in this local CPU environment.
