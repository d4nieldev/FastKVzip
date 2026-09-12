# LongBench Evaluation Integration

## Summary

Add all **21 original LongBench tasks**, including Chinese and code tasks, shared across GraphKV, FastKVzip, and KVzip. Exclude LongBench-E and LongBench v2.

Use official task instructions, output limits, and scoring, while retaining our existing model/chat wrapping and GraphKV checkpoint prefix, as selected. This is a method comparison using LongBench—not an exact reproduction of its upstream prompting protocol.

## Branch and data integration

- Create branch `feature/longbench-evaluation` and worktree `.worktrees/longbench-evaluation` from freshly fetched `origin/main`, preserving the dirty workspace.
- Save this plan in `docs/plans/longbench-evaluation.md`.
- Support `--data longbench` and individual selectors such as `--data longbench_qasper` and `--data longbench_repobench-p`.
- Expand `--data all` from 107 to 128 benchmarks; continue excluding Agentic. Keep the historical RULER grid and coordinator explicitly scoped to their original 107 benchmarks.
- Download the official [Hugging Face archive](https://huggingface.co/datasets/zai-org/LongBench/tree/5e628be450b7e67fb7ae6e201bd6d8f7056f7672) through standard HF caching, pinned to that revision. Read the selected JSONL member directly using `zipfile`; no preparation command, extraction directory, remote dataset script, or new cache flag.
- Return the existing context/question/answers structure, preserving raw context, demonstrations, whitespace, and all accepted references. Preserve full benchmark size when applying existing limits/ranges. Evaluate complete published test splits by default.

## Prompts, generation, and scoring

- Split each official task template around `{context}`. Keep preceding instructions protected from compression; feed the formatted trailing question or completion request after compressed prefill.
- Append protected instructions to the effective prefix **after checkpoint-prefix restoration**, without accumulating changes across examples or retention ratios.
- Preserve exact task suffix formatting, bypassing generic `Q:` insertion and whitespace stripping. Retain existing chat boundaries for every task.
- Keep existing greedy generation and EOS handling, applying official per-task output caps. Do not introduce raw-completion modes or SAMSum-specific stopping behavior.
- Adapt scoring from the [pinned official implementation](https://github.com/THUDM/LongBench/tree/2e00731f8d0bff23dc4325161044d0ed8af94c1e/LongBench): English/Chinese QA F1, ROUGE-L, classification, retrieval, counting, and code similarity. Apply official prediction postprocessing and maximum score across accepted references.
- Reuse existing scoring dependencies; add only `jieba` for Chinese segmentation. Keep LongBench normalization separate because existing QA normalization changes number words.
- Supply TREC and LSHT class metadata through the existing supplementary-reference loader. No adapter registry or result-schema redesign.

## Compatibility and verification

- Reuse all three existing evaluation entry points, native compression procedures, result storage, and W&B uploader.
- Bind LongBench dataset and protocol revisions into existing manifests. Reject incompatible resumes while leaving non-LongBench result identities unchanged.
- Test selectors, all 21 conversions, ranges/full sizes, reference preservation, protected prompt boundaries, code whitespace, output caps, and official metric parity.
- Test repeated prefill without prefix accumulation, classification metadata alignment, resume compatibility, and identical online/offline scoring.
- Exercise GraphKV—including its chunked runner—FastKVzip, and KVzip through CPU test doubles. Preserve existing SCBench/RULER regression tests and historical 107-job inventory checks.
- Run focused tests, the complete prefill test suite, and compilation checks.

## Delivery boundaries

Commit verified changes, push the dedicated branch, and open a PR to `main` with protocol documentation and evaluation examples.

This change adds evaluation support only: no retraining, cluster submissions, W&B writes, or paper updates. A production LongBench grid will be proposed separately.
