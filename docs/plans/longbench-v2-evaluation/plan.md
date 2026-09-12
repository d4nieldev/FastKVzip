# LongBench v2 — approved plan

The proposal and approval below are preserved verbatim from the conversation.
The later branch agreement supersedes the earlier suggestion to extend PR #24.

## Initial implementation scope

I can keep this small by extending [PR #24](https://github.com/d4nieldev/FastKVzip/pull/24):

- Add `--data longbench_v2`, also included in `all`.
- Reuse all three evaluators, caching, prompt handling, result storage, and uploads.
- Add the pinned dataset, official multiple-choice prompt, 128-token output cap, accuracy scorer, and focused tests.

## Clarified proposal

Agreed. I’ll create `feature/longbench-v2-evaluation` from `feature/longbench-evaluation`, in its own worktree, and open the PR **against the LongBench branch**. The existing PR will remain unchanged.

### 1. Direct-answer evaluation

The official evaluator supports two distinct protocols:

| Protocol | What happens |
|---|---|
| Direct-answer—the default | One request containing context, question, and four choices; up to **128 output tokens**. |
| CoT—optional | First generate up to **1,024 reasoning tokens**. Then make a separate request containing the question, choices, and generated reasoning—but **not the original context**—to produce the final answer within 128 tokens. |

Sources: [official evaluator](https://github.com/THUDM/LongBench/blob/2e00731f8d0bff23dc4325161044d0ed8af94c1e/pred.py), [second-pass CoT prompt](https://github.com/THUDM/LongBench/blob/2e00731f8d0bff23dc4325161044d0ed8af94c1e/prompts/0shot_cot_ans.txt).

My recommendation is to implement **the official direct-answer setting first**. It fits our current one-pass evaluation: prefill, compress, ask the question, generate, score. We would use its official task prompt and answer extraction/accuracy.

This is not an invented simplification of the benchmark—it is one of its official settings. However, it does **not** measure the benefit of its longer CoT protocol, so we should compare against direct-answer results, not CoT results.

CoT would require more than a different prompt: a second generation workflow, storing intermediate reasoning, and resolving our existing prefix’s “without explanation” instruction.

### 2. Truncating overlong contexts

The official code already truncates overlong inputs. It tokenizes the **entire formatted prompt**, keeps its first and last halves within a configured budget, and discards the middle. Its configuration uses **120,000 tokens for ordinary Qwen2.5-7B-Instruct**; our newer 1M checkpoint is not listed. [Truncation code](https://github.com/THUDM/LongBench/blob/2e00731f8d0bff23dc4325161044d0ed8af94c1e/pred.py#L20-L32), [configured limits](https://github.com/THUDM/LongBench/blob/2e00731f8d0bff23dc4325161044d0ed8af94c1e/config/model2maxlen.json).

My proposal follows that approach, with a deliberate adaptation:

- Truncate **only the context**, preserving the complete instructions, question, and choices.
- Reserve space for generation and KVzip’s reconstruction-scoring inputs.
- Give all three methods the **same truncated context**.
- Apply retention ratios afterward, to that context.

For example, at 20% retention, we retain 20% of the context remaining **after truncation**, not 20% of the original document.

This can remove relevant evidence, just as upstream truncation can. Also, using our model’s roughly 1M capacity instead of upstream’s 120K setting changes how much evidence is available. Neither that capacity nor truncation guarantees the evaluation fits a particular GPU.

### How closely would we match upstream?

We would match its **direct-answer task instructions, output cap, and scoring**. We would retain our existing chat/checkpoint prefix and greedy generation; upstream requests temperature `0.1`. Context-only truncation and the larger input budget would also be documented differences.

My recommendation remains direct-answer first, with shared beginning/end truncation: a small integration for comparing our three methods, **not an exact reproduction of upstream leaderboard settings**.

## User approval

1. I agree, lets do official direct answer setting
2. I agree, also lets use 1M capacity instead of 120K
3. About the temperature - we should keep all our eveluations consistent, which I think we used temperature 0 (greedy generation) - please correct if if I'm wrong
