# Batch stage-2 subgraphs using the token microbatch budget

## Summary

Make stage-2 initial scoring, validation, and backward replay process multiple independent subgraphs together.

With `--token-microbatch-size 16000` and `--subgraph-size 2000`, each scorer call processes up to **eight subgraphs per layer/head graph**, subject to `--graph-microbatch-size`.

## Implementation

- Extract the standalone evaluator’s existing per-group packing and scoring into a differentiable helper in the graph model module. Share it between stage 2 and standalone fixed-subgraph evaluation.
- Use the existing subgraph grouping utility. Pack equal-length subgraphs into `[subgraph_count × graph_microbatch, subgraph_size, hidden_dim]`, repeating their layer/head identities.
- Preserve independent connectivity, Gram matrices, and normalization for each subgraph. Restore scores to their original layer/head and token order.
- Process an incomplete group at its actual size. Score a shorter final subgraph separately, without padding.
- Use identical grouping in initial scoring and replay. During replay, backpropagate each token-group/graph-microbatch pair immediately using the corresponding external score gradients, then release its activations.
- Sum replay gradients without additional averaging. Gradient accumulation across questions continues to provide the existing equal-question weighting.
- Slice and materialize only the current microbatch’s hidden states during replay, keeping saved scorer activations bounded by both microbatch settings.

## Interfaces and behavior

- Use the existing flags and defaults. Retain the requirement that token microbatch be divisible by the configured subgraph size.
- A token microbatch equal to subgraph size processes one subgraph at a time. Whole-context graph mode retains its existing streamed-token behavior.
- Prefill still completes before graph scoring. Global top-k selection still operates over the complete context after scores are concatenated.
- Preserve checkpoint fields, existing resume-setting checks, optimizer cadence, schedules, and metric meanings.
- Document that token and graph microbatches multiply: eight subgraphs with graph microbatch 112 produce up to **896 graph instances** per scorer call and require more activation memory.
- Validate mathematical equivalence with numerical tolerances; do not require identical GPU trajectories across different batching shapes.

## Validation and delivery

- Compare packed scores against explicit sequential scorer calls using multiple layers/heads, eight-subgraph groups, partial groups, short tails, and a partial graph microbatch. Verify that changing one subgraph does not affect another’s scores.
- Compare every gate/mixer parameter gradient against an independent sequential reference using nonuniform external gradients.
- Extend replay-memory tests to verify that saved activations remain bounded as context length grows and that inference-mode inputs do not cause full-context cloning.
- Exercise real answer-loss training with grouped subgraphs, unequal answer lengths, question accumulation, frozen LLM weights, and normal stop/resume. Verify parameters, counters, schedules, and logged metrics.
- Run the relevant graph-model, evaluation, graph-training, answer-training, batching, and checkpoint regression suites.
- Implement in a new worktree and feature branch from current `origin/main`. Save this approved plan and subsequent decisions under `docs/changes/stage2-subgraph-batching/`, update stage-2 documentation, commit, push, and open one PR.
- Preserve unrelated working-tree changes. Cluster benchmarking and training submission remain separate tasks.
