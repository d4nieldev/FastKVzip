# Whole-Context Graph FastKVzip: Pipeline and Decision Audit

This document explains the implementation in pull request #1 at code commit
`1d2af0c2e130bb19f4e1249a393c56f283aadee1`.

Its scope includes the new graph code and the old FastKVzip code reused by the
new pipeline. A decision is included when it changes model meaning, gradients,
memory, reproducibility, compatibility, evaluation, or reviewability. Small
input checks and test-only plumbing are omitted unless they materially define
the public runner interface or failure behavior.

## How to read the origin lines

Each decision cites its strongest source:

1. **User formulation** means you proposed the decision yourself.
2. **User agreement** means the decision was proposed in discussion and you
   explicitly accepted it.
3. **Approved plan** means it appeared in the implementation plan but did not
   have a stronger conversation source.
4. **Inherited behavior** means the graph code deliberately keeps behavior
   from the original repository.
5. **Implementation-only** means the code chose something we never discussed.

Conversation quotations are copied from this thread. Plan quotations name the
section in the approved “Whole-Context Graph FastKVzip” plan. Every decision
also links to its current implementation.

## Reading map

Read the opening pipeline first, then use these sections as a review checklist:

1. [Scope and integration](#1-scope-and-integration-decisions)
2. [Prefix, hidden states, and teacher labels](#2-prefix-hidden-state-and-teacher-label-decisions)
3. [Graph model](#3-graph-model-decisions)
4. [FAISS topology](#4-faiss-topology-decisions)
5. [Training and gradients](#5-training-and-gradient-decisions)
6. [Data, optimization, checkpoints, and logging](#6-data-order-optimization-checkpoints-and-logging)
7. [Evaluation and pruning](#7-evaluation-and-pruning-decisions)
8. [Object lifetime and gradients](#8-object-lifetime-device-and-gradient-map)
9. [Verification evidence](#9-verification-contract-and-current-evidence)
10. [Known limitations](#10-known-limitations-and-work-still-requiring-a-real-pilot)
11. [Audited source map](#11-audited-source-map)

## Review these implementation-only choices first

These choices have the largest effect and were not explicitly approved:

- **F1:** FAISS runs on CPU and builds graphs sequentially inside a graph
  microbatch.
- **F4:** Every IVF index is trained from scratch for every graph construction.
- **M9:** A future learnable builder receives no layer/head IDs, so its natural
  parameterization is shared across graphs.
- **M12:** With zero `B`, the first context gives no task-loss gradient to `A`
  or GIN.
- **T4:** Graph/joint processing still keeps full-context `z` and `u` on GPU;
  gate token updates instead reuse a complete CPU `u` cache.
- **O8:** Resume checks omit weight decay, seed, epoch target, dataset version,
  library versions, and Git commit.
- **E6:** Only prefix IDs are restored; postfix IDs come from the current code.
- **E11:** The local window gets maximum scores, but it is not an absolute keep
  mask.
- **O14:** Training and validation graph BCE use the same W&B metric name.
- **S4:** Raw CLI defaults use a random gate and online W&B, while the README’s
  recommended command supplies `fastkvzip` and offline W&B.

These are records, not automatic recommendations to change the code.

## Symbols and exact shapes

| Symbol | Meaning |
|---|---|
| `P` | Number of prefix tokens. |
| `T` | Number of context tokens. |
| `L` | Number of transformer layers. |
| `H` | Number of KV heads per layer. |
| `G=L×H` | Number of token graphs for one context. |
| `D` | LLM hidden dimension. |
| `R` | Graph node dimension, called `graph_dim`. |
| `Q` | Number of query heads sharing one KV head. |
| `M` | Number of complete graphs in one graph microbatch. |
| `k` | Requested neighbors per token. |

The main tensors are:

```text
Captured hidden cache       L tensors, each [1, P+T, D] on CPU
Teacher labels              [L, 1, H, T]
Training hidden states      L tensors, each [T, D] on CPU
Graph hidden microbatch     [M, T, D]
Projected nodes z           [M, T, R]
Propagated nodes u           [M, T, R]
Residual delta              [M, T, D]
Graph scores                [L, 1, H, T]
Hard-FAISS edge index       [2, M×T×min(k,T-1)]
```

`gate_sink` and cache `sink` are different concepts. `gate_sink` is the number
of learned FastKVzip base keys. Cache `sink` is the prefix length `P` protected
from pruning.

## The four different kinds of chunking

These controls do different jobs:

1. **Prefill chunks default to 16,000 tokens.** They limit temporary LLM
   prefill memory while later chunks attend the earlier KV cache.
2. **Teacher reconstruction chunks are fixed at 2,000 tokens.** They define the
   pieces for which original KVzip reconstruction scores are generated.
3. **Token microbatches default to 1,000 positions.** They limit hidden transfer
   and tokenwise gate/loss work. They never split the GNN graph.
4. **Graph microbatches count layer/head graphs, not tokens.** `auto` means `H`
   complete graphs, normally all heads of one layer.

## End-to-end pipeline

### Training

1. `train_graph.py` validates model-independent options.
2. It starts or resumes a W&B run.
3. It loads `ModelKVzip` with a retain cache and no built-in gate.
4. `DataWrapper` tokenizes one context and installs the model/task prefix.
5. The LLM prefills `[prefix, context]` in KV-cached chunks.
6. Attention inputs for every layer are copied to CPU during that prefill.
7. Original KVzip reconstruction scoring creates one target per
   layer/KV-head/context-token.
8. Only context hidden states, context IDs, labels, and metadata are copied into
   one in-memory `TeacherExample`.
9. The large teacher KV cache is deleted.
10. The student creates one graph for each `(layer, KV head)` pair.
11. `A` projects hidden states, FAISS chooses neighbors, and GIN mixes nodes.
12. `B` maps the mixed node back to hidden width.
13. The existing FastKVzip gate scores `x + B(u)`.
14. Two-phase mode first updates the gate in token batches and then updates the
    graph side once. Joint mode updates both sides once.
15. Validation runs after the fixed training split for each epoch.
16. The trainer calls W&B once, then advances the cursor and saves `last.pt`
    after every successfully processed context.
17. `best.pt` is replaced after a full validation pass improves mean graph BCE.

### Evaluation

1. `eval_graph.py` loads and validates the checkpoint on CPU.
2. It reconstructs the exact graph scorer and an ungated retain-cache LLM.
3. It restores the training prefix and saved prefill chunk size.
4. It prefills `[prefix, context]` and captures the same layer inputs on CPU.
5. It graph-scores only the context range and assigns `[L,1,H,T]` to `kv.score`.
6. It raises scores for the existing local context window.
7. Once scoring begins, it clears all captured hidden states on success or
   scoring failure.
8. It generates the full-cache reference answer.
9. For each existing retention ratio, it builds a context mask and regenerates
   with the pruned attention view.
10. Prefix, question, postfix, and generated tokens stay outside the scored
    region and are retained by cache-mask padding.
11. Existing result helpers save one JSON result per sample and prune level.

## 1. Scope and integration decisions

### S1. The graph path is additive and leaves the old training/evaluation paths unchanged.

- **Origin:** **User formulation** — “What if we have a brand new script? Can
  we reuse relevant methods and keep it relatively slim?”
- **Rationale:** The graph experiment can be reviewed and run without changing
  baseline behavior.
- **Alternatives:** Modify `eval.py` and the offline feature pipeline, or refactor
  both paths into one shared runner.
- **Code:** [`train_graph.py`](../prefill/train_graph.py#L770-L908),
  [`eval_graph.py`](../prefill/eval_graph.py#L49-L124).

### S2. Training handles one complete context at a time and creates no dataset artifacts.

- **Origin:** **User formulation** — “we can actually produce the dataset
  on-the-fly ... then we don't care about it anymore.”
- **Rationale:** It avoids storing very large activations, scores, and graph
  edges on the cluster filesystem.
- **Alternatives:** Build an offline dataset, memory-map activations, or batch
  several contexts together.
- **Code:** [`run_training`](../prefill/train_graph.py#L825-L897),
  [`TeacherExample`](../prefill/graph/training.py#L43-L73).

### S3. The current graph path supports ordinary decoder hidden caches, not hybrid/static layouts.

- **Origin:** **Implementation-only.**
- **Rationale:** The implementation assumes exactly one captured tensor for
  every transformer layer.
- **Alternatives:** Add a model-family layer map, score only static layers, or
  create cache-layout adapters.
- **Code:** [`_validate_hidden_cache`](../prefill/graph/evaluation.py#L307-L346),
  [README limitation](../prefill/README.md#L34-L36).

### S4. Raw CLI defaults differ from the README's recommended workflow.

- **Origin:** **Implementation-only.**
- **Rationale:** The CLI keeps “no checkpoint” available as the random-gate
  experiment, while the README demonstrates the expected released-gate run.
- **Alternatives:** Default `--gate-checkpoint` to `fastkvzip`, default W&B to
  offline, or describe the raw and recommended defaults separately.
- **Code:** [parser defaults](../prefill/train_graph.py#L52-L103),
  [README command](../prefill/README.md#L38-L46).

### S5. FAISS, PyG, and W&B are hard runtime dependencies.

- **Origin:** **Implementation-only.**
- **Rationale:** Imports stay simple. Requirements pin CPU FAISS 1.15.0 and PyG
  2.8.0.post1, while W&B has a lower bound of 0.19 rather than an exact pin.
- **Alternatives:** Optional extras, lazy imports, GPU FAISS, or a Torch-only
  exact builder.
- **Code:** [`requirements.txt`](../prefill/requirements.txt#L1-L24),
  [builder imports](../prefill/graph/builder.py#L6-L10).

### S6. Commands are run from `prefill/` on one CUDA device.

- **Origin:** **Implementation-only.**
- **Rationale:** The original repository uses local imports, relative result
  paths, CUDA attention, and a single model device.
- **Alternatives:** Package entry points, root-independent paths, CPU fallback,
  or explicit model-parallel support.
- **Code:** [README](../prefill/README.md#L34-L36),
  [`GraphScorer` device choice](../prefill/graph/model.py#L289-L326).

## 2. Prefix, hidden-state, and teacher-label decisions

### D1. Training and evaluation use the exact same saved prefix before the context.

- **Origin:** **User formulation** — “the hidden states for the context are
  generated exactly the same in training and evaluation ... if we change the
  system prompt or prefix in evaluation things might not work the same.”
- **Rationale:** A context token's hidden state depends on all earlier prefix
  tokens.
- **Alternatives:** Recompute the current template prefix, or train with several
  prefixes for robustness.
- **Code:** [prefix checkpointing](../prefill/graph/training.py#L299-L351),
  [prefix restoration](../prefill/graph/evaluation.py#L502-L505).

### D2. Only context positions are labels and graph nodes.

- **Origin:** **User formulation** — “append the question (while evicting only
  kv cache values of the context token positions)?”
- **Rationale:** Prefix/start-turn tokens occur before `start_idx`; question,
  postfix, and generated tokens occur after `end_idx`.
- **Alternatives:** Include prefix nodes but discard their scores, or learn
  separate scores for every region.
- **Code:** [eviction range](../prefill/model/wrapper.py#L170-L175),
  [training slice](../prefill/train_graph.py#L427-L454).

### D3. Hidden states are captured at each attention layer's input during normal causal prefill.

- **Origin:** **Approved plan, Online Training** — “Prefill using the existing
  chunked KV-cache path.” Capturing the attention-layer input is **inherited
  behavior** from the repository's `save_hidden` seam.
- **Rationale:** These are the exact hidden vectors consumed by the unchanged
  FastKVzip gate; later chunks attend earlier cached KV.
- **Alternatives:** Capture block outputs, K/V vectors, or separately prefill
  independent context chunks.
- **Code:** [capture point](../prefill/attention/attn.py#L24-L65),
  [chunked prefill](../prefill/model/wrapper.py#L155-L227).

### D4. Prefill defaults to 16,000 tokens and the chosen size is checkpointed.

- **Origin:** **Implementation-only default; approved-plan persistence.** The
  plan requires checkpoints to contain “exact prefix token IDs and prefill
  chunk size.”
- **Rationale:** It follows the repository's existing long-context default and
  makes evaluation reproduce the same call boundaries.
- **Alternatives:** Infer from memory, use one full forward, or expose separate
  training/evaluation sizes.
- **Code:** [default](../prefill/train_graph.py#L367-L378),
  [saved value](../prefill/graph/training.py#L345-L351).

### D5. The checkpoint saves prefix IDs but not postfix IDs.

- **Origin:** **Implementation-only.**
- **Rationale:** Postfix tokens occur after the context and cannot change the
  captured context hidden states. They do affect reconstruction prompts and
  teacher labels, as well as evaluation queries and generation.
- **Alternatives:** Save both prefix and postfix for complete prompt-template
  reproduction, or save the entire serialized template.
- **Code:** [checkpoint payload](../prefill/graph/training.py#L345-L351),
  [reconstruction prompt](../prefill/model/wrapper.py#L229-L269),
  [prefix-only restore](../prefill/graph/evaluation.py#L502-L505).

Saving only the prefix reproduces context activations. It does not snapshot the
complete training-target or evaluation token sequence.

### D6. The first generated example defines the training prefix, and every later example must match it.

- **Origin:** **Implementation-only.**
- **Rationale:** It catches a dataset wrapper silently changing the hidden-state
  distribution inside one run.
- **Alternatives:** Fix the prefix before loading any dataset, allow per-dataset
  prefixes, or store a prefix with each example.
- **Code:** [`make_example`](../prefill/train_graph.py#L827-L847).

### D7. The teacher LLM has no built-in gate, so labels are reconstruction scores rather than released-gate predictions.

- **Origin:** **Approved plan, Online Training** — “Load the teacher with
  `gate_path_or_name=""`.”
- **Rationale:** This avoids the original failure mode where data generation
  accidentally used FastKVzip gate outputs as its own labels.
- **Alternatives:** Distill the released gate deliberately, or add an explicit
  teacher-label mode.
- **Code:** [`build_teacher`](../prefill/train_graph.py#L419-L425),
  [legacy score switch](../prefill/model/wrapper.py#L217-L225).

### D8. Teacher reconstruction scoring uses fixed, non-overlapping 2,000-token target chunks.

- **Origin:** **Inherited behavior.**
- **Rationale:** It reuses KVzip's reconstruction task and bounds temporary
  query-side memory. For later chunks, eight tokens from the previous chunk are
  included as a continuation cue, not as target overlap.
- **Alternatives:** Make the chunk size configurable, overlap target chunks, or
  reconstruct the entire context in one call.
- **Code:** [`self_task`](../prefill/model/wrapper.py#L229-L255),
  [`scoring`](../prefill/model/wrapper.py#L257-L274).

### D9. Each teacher label is the maximum reconstruction attention over query positions and query groups.

- **Origin:** **Inherited behavior.**
- **Rationale:** It preserves the original continuous KVzip importance target
  for every layer, KV head, and token.
- **Alternatives:** Mean attention, sum attention, output reconstruction error,
  or a ranking target.
- **Code:** [`KVScore._get_score`](../prefill/attention/score.py#L40-L72).

### D10. Teacher labels are continuous probabilities trained with ordinary BCE.

- **Origin:** **User formulation** — “using the kvzip scores for the BCE loss.”
- **Rationale:** BCE can distill a target in `[0,1]`; labels do not need to be
  hard zero/one classes.
- **Alternatives:** MSE, KL divergence, pairwise ranking, or binarized labels.
- **Code:** [`_bce_sum`](../prefill/graph/training.py#L544-L553).

### D11. A `TeacherExample` owns CPU clones of hidden states, labels, IDs, and prefix metadata.

- **Origin:** **User formulation for context identity** — “In teacher data
  generation, we will also need to save context id information for training.”
  CPU hidden-state storage comes from the **approved plan, Online Training**;
  clone ownership is **implementation-only**.
- **Rationale:** The in-memory example remains valid after the KV cache is
  deleted and preserves dataset identity.
- **Alternatives:** Keep views into the cache, retain only hidden/labels, or
  persist examples to disk.
- **Code:** [`TeacherExample`](../prefill/graph/training.py#L32-L73),
  [example conversion](../prefill/train_graph.py#L427-L454).

The current trainer does not use `token_ids`, `dataset_name`, or
`dataset_index` in its loss. One context is already isolated, so those fields
are metadata rather than graph-grouping inputs.

### D12. The teacher KV cache is deleted before student training, without forcing `torch.cuda.empty_cache()`.

- **Origin:** **Approved plan, Online Training** — “Delete the KV cache before
  student training.”
- **Rationale:** Live KV tensors can be reclaimed while PyTorch's allocator is
  allowed to reuse reserved blocks efficiently.
- **Alternatives:** Explicitly empty the CUDA allocator, overlap teacher and
  student work, or retain the cache for debugging.
- **Code:** [`make_example`](../prefill/train_graph.py#L835-L847).

## 3. Graph-model decisions

### M1. One context contains `L×H` independent graphs, ordered layer first and KV head second.

- **Origin:** **User formulation for graph identity** — “for each kv head in
  each layer we will construct a token-graph.” Layer-major/head-minor ordering
  is **implementation-only**.
- **Rationale:** Each KV head can learn different neighbors and transformations;
  layer-major IDs make the default microbatch one complete layer.
- **Alternatives:** One graph per layer, one graph for the whole model, shared
  head parameters, or head-major ordering.
- **Code:** [graph count and IDs](../prefill/graph/model.py#L265-L268),
  [`graph_batches`](../prefill/graph/model.py#L338-L354).

### M2. Gate and graph widths are separate hyperparameters with defaults 16 and 32.

- **Origin:** **User formulation** — “make the graph node embedding dimension
  and the latent queries and keys not the same hyperparameter ... default graph
  dim ... 32.” `gate_dim=16` comes from the **approved plan, Architecture**.
- **Rationale:** Graph capacity and FastKVzip gate capacity can be tuned
  independently.
- **Alternatives:** Tie both widths, derive graph width from hidden size, or use
  model-specific defaults.
- **Code:** [CLI resolution](../prefill/train_graph.py#L262-L268),
  [`GraphScorer`](../prefill/graph/model.py#L252-L263).

`gate_dim=16` is only a training default. `GraphScorer` accepts the width of a
loaded custom or released gate.

### M3. `A`, `B`, and GIN parameters are independent for every layer/head graph.

- **Origin:** **Approved plan, Architecture** — “Independent `A`, `B`, and GIN
  parameters per layer/head.”
- **Rationale:** No graph is forced to use another layer or head's feature
  transform.
- **Alternatives:** Share by layer, share by head index, use low-rank shared
  factors, or use one global GNN.
- **Code:** [`PerGraphLinear`](../prefill/graph/model.py#L110-L139),
  [`GroupedGIN`](../prefill/graph/model.py#L142-L175).

The parameter cost of `A` and `B` alone is `2×L×H×D×R` scalars.

### M4. `A` and `B` have no bias; GIN's two linear layers do have bias.

- **Origin:** **Approved plan, Architecture** — “Bias-free `A` and `B`.” The
  GIN bias choice is **implementation-only**.
- **Rationale:** `A/B` implement pure per-graph projections, while GIN uses
  ordinary PyTorch `Linear → ReLU → Linear` modules.
- **Alternatives:** Add `A/B` biases, remove GIN biases, or make all bias choices
  configurable.
- **Code:** [`PerGraphLinear`](../prefill/graph/model.py#L123-L139),
  [GIN MLP construction](../prefill/graph/model.py#L161-L175).

### M5. GIN uses `Linear → ReLU → Linear` after `(1+epsilon)×self + neighbor_sum`.

- **Origin:** **Approved plan, Architecture** — “GIN MLP: `Linear → ReLU →
  Linear`” and “Learnable GIN epsilon.”
- **Rationale:** This is the canonical GIN update; explicit self information
  remains even though FAISS self-edges are removed.
- **Alternatives:** Mean/max aggregation, GCN normalization, attention, fixed
  epsilon, or explicit self-edges.
- **Code:** [GIN update](../prefill/graph/model.py#L177-L212).

Epsilon starts at zero. Aggregation is implemented with Torch `index_add`;
PyG supplies `EdgeIndex`, not `GINConv`.

### M6. One topology is reused across all configured GIN depths.

- **Origin:** **Implementation-only.**
- **Rationale:** Rebuilding a FAISS index after every GIN layer would multiply
  graph-construction cost.
- **Alternatives:** Recompute neighbors from each GIN layer, learn a topology per
  depth, or restrict depth permanently to one.
- **Code:** [single builder call](../prefill/graph/model.py#L364-L369),
  [depth loop](../prefill/graph/model.py#L195-L211).

### M7. `B(u)` is added to `x` as an unscaled residual before both gate projections.

- **Origin:** **Approved plan, Architecture** — `score[l,h] =
  FastKVzipHead[l,h](x[l] + delta[l,h])`.
- **Rationale:** It preserves the original hidden width and lets the unchanged
  gate consume a normal hidden-state tensor.
- **Alternatives:** Concatenate features, use a learned residual coefficient,
  normalize/clip `delta`, or apply it only to the gate's query or key branch.
- **Code:** [mixing and gate projections](../prefill/graph/model.py#L218-L245),
  [`B`](../prefill/graph/model.py#L388-L406).

The same head-specific `delta` is used for all query groups attached to that KV
head.

### M8. The old gate stays unchanged; a private adapter selects one head's parameter slices.

- **Origin:** **Approved plan, Architecture** — “The existing FastKVzip gate
  remains unchanged. An internal `HeadwiseGateAdapter` applies each
  head-specific mixed input.”
- **Rationale:** Different heads need different `x+delta`, which cannot be
  represented by one call to the old full-head gate.
- **Alternatives:** Add `forward_head()` to `Weight`, instantiate one gate per
  head, or formalize a new public gate interface.
- **Code:** [`_HeadwiseGateAdapter`](../prefill/graph/model.py#L215-L246).

The adapter is coupled to the concrete `Weight` fields. A different gate layout
is not automatically supported.

It projects the already-added `x+delta` once. Projecting `x` and `delta`
separately and then adding would be algebraically similar in exact arithmetic,
but would not preserve the tested BF16 operation order and bit parity.
See the [distinct-delta parity test](../prefill/tests/test_graph_model.py#L262-L285).

### M9. The graph-builder seam is an `nn.Module` with `forward(z,k)` and optional differentiable edge weights.

- **Origin:** **User formulation for learnability** — “we should also support
  graph builders that have learnable weights.” The `GraphTopology`,
  `forward(z,k)`, and optional `edge_weight` contract comes from the **approved
  plan, Graph Construction and Parallelism**.
- **Rationale:** Builder parameters register normally, move with the scorer, enter
  the graph optimizer, and can receive gradients through `edge_weight`.
- **Alternatives:** A plain callable, a PyG `Data` return value, or a builder that
  returns already-propagated nodes.
- **Code:** [`GraphBuilder` and `GraphTopology`](../prefill/graph/builder.py#L13-L24),
  [weighted consumption](../prefill/graph/model.py#L194-L201).

Builder parameters register in the graph optimizer and graph checkpoint through
normal `nn.Module` traversal.

**Important limitation:** `forward(z,k)` receives no global graph IDs. A future
learnable builder naturally shares its parameters across layer/head graphs. A
builder needing independent layer/head behavior would normally need graph
identity through some additional seam or separate builder modules.

### M10. The scorer's standard forward handles one context and returns `[L,1,H,T]`.

- **Origin:** **User formulation for the whole-context input** — “a training
  example is all token positions in layer `i` and kv-head `j` ...
  `[seq_len, hidden_dim]`.” The output shape is from the **approved plan,
  Standalone Evaluation** and inherited pruning compatibility.
- **Rationale:** It matches the singleton-batch score layout expected by the old
  pruning code.
- **Alternatives:** Add a context batch dimension or expose only flat
  `[L×H,T]` scores.
- **Code:** [`GraphScorer.forward`](../prefill/graph/model.py#L408-L451).

Training and evaluation normally call its lower-level projection, propagation,
and scoring methods to stage memory. The full `forward` is the reference path.

### M11. Low-precision activations use FP32 master parameters.

- **Origin:** **Implementation-only.**
- **Rationale:** Direct BF16/FP16 AdamW updates at the configured learning rates
  can round to zero. FP32 masters preserve small updates while `z`, `u`, and
  gate math remain in the LLM's compute dtype.
- **Alternatives:** Native low-precision parameters, AMP with an external master
  copy, or full FP32 computation.
- **Code:** [master dtype selection](../prefill/graph/model.py#L273-L326),
  [operation-time casts](../prefill/graph/model.py#L356-L406).

Only float16, bfloat16, float32, and float64 compute dtypes are supported. The
implementation assumes one device and one compute dtype for all layers.

### M12. Automatic `B` initialization is zero with a gate checkpoint and random without one.

- **Origin:** **Approved plan, Initialization and Checkpoints** — “checkpoint
  present: `auto → zero`; no checkpoint: `auto → random`.”
- **Rationale:** Zero `B` reproduces the loaded FastKVzip gate exactly; random
  `B` prevents a completely fresh graph path from beginning closed.
- **Alternatives:** Always zero, always small random, or a learned scalar opened
  gradually.
- **Code:** [`resolve_b_init`](../prefill/graph/training.py#L167-L183),
  [component setup](../prefill/train_graph.py#L722-L731).

This rule applies only to a fresh run with `--b-init auto`. An explicit
`--b-init zero` or `random` overrides it. Resume restores the saved `B` and does
not initialize it again.

**Consequence not discussed earlier:** when `B=0`, the first context's
task-loss gradient reaches `B` but not `A` or GIN. `A` and GIN receive task
signal only after `B` becomes nonzero on a later context. Decoupled weight decay
may still move parameters with zero gradients.

### M13. Delta energy share is a metric, not a loss or constraint.

- **Origin:** **User formulation for the measurement** — “one of the metrics I
  want logged ... how much `B` is giving more weight to the mixed input
  embeddings.” The exact bounded formula comes from the **approved plan, W&B
  Logging**.
- **Rationale:** `sum(delta²)/(sum(x²)+sum(delta²))` is bounded in `[0,1]` and
  cheap to accumulate.
- **Alternatives:** Delta/hidden norm ratio, cosine change, per-layer metrics,
  or a regularizer that actively limits the residual.
- **Code:** [model accumulation](../prefill/graph/model.py#L421-L447),
  [energy-share formula](../prefill/graph/training.py#L568-L575),
  [gate accumulation](../prefill/graph/training.py#L627-L629),
  [graph accumulation](../prefill/graph/training.py#L699-L701).

The sum uses FP32 when compute is FP16/BF16 to avoid overflow. Hidden energy is
counted once per graph/head, matching the replicated deltas.

### M14. GIN has no dropout, graph normalization, or cross-graph mixing.

- **Origin:** **Approved plan, Architecture** — “No dropout or cross-graph
  normalization.”
- **Rationale:** Each layer/head graph remains independent and the smallest GIN
  stays easy to compare with FastKVzip.
- **Alternatives:** LayerNorm/GraphNorm, dropout, shared global nodes, or
  cross-head message passing.
- **Code:** [GIN modules](../prefill/graph/model.py#L142-L175),
  [block-diagonal graph layout](../prefill/graph/builder.py#L121-L145).

### M15. The default graph uses 16 neighbors and one GIN layer.

- **Origin:** **Approved plan, Architecture** — `gin_depth=1` and
  `num_neighbors=16`.
- **Rationale:** This is a shallow one-layer baseline with a nontrivial
  16-neighbor neighborhood before the FastKVzip gate.
- **Alternatives:** Deeper GIN, a smaller/larger neighborhood, or defaults that
  scale with context length.
- **Code:** [default resolution](../prefill/train_graph.py#L266-L270),
  [scorer construction](../prefill/train_graph.py#L701-L721).

### M16. The complete student path is `x → A → topology → GIN → B → gate(x+delta)`.

- **Origin:** **User formulation** — “feed this graph into some GNN to mix
  features, then ... send the mixed features into their gate architecture.” The
  exact `A/B` residual path comes from the **approved plan, Architecture**.
- **Rationale:** Topology is selected from `z=A(x)`; GIN consumes the same live
  `z`; and `B(u)` changes only the input to the unchanged FastKVzip gate.
- **Alternatives:** Build topology from raw `x`, detach `z` before GIN,
  concatenate graph features, or replace the gate with a new scoring head.
- **Code:** [projection and propagation](../prefill/graph/model.py#L356-L369),
  [residual and scoring](../prefill/graph/model.py#L388-L406).

### M17. Independent graph networks execute in grouped batches, while headwise gate calls still loop in Python.

- **Origin:** **User formulation for parallelism** — “we can parallelize over
  layers and heads.” PyG `EdgeIndex` and grouped GIN come from the **approved
  plan, Graph Construction and Parallelism**; the exact execution split is
  **implementation-only**.
- **Rationale:** `A`, `B`, and each GIN MLP use selected parameter rows and
  batched matrix multiplications over `M` graphs. The adapter still invokes the
  matching gate head one graph at a time.
- **Alternatives:** Vectorize the headwise gate too, use independent PyG
  `Data` objects, or loop over every graph for all modules.
- **Code:** [batched linear helper](../prefill/graph/model.py#L82-L96),
  [grouped GIN](../prefill/graph/model.py#L142-L212),
  [headwise loop](../prefill/graph/model.py#L371-L386).

The implementation preserves per-graph nested GIN parameter keys for strict
checkpoint loading even though execution is grouped. See the
[layout compatibility test](../prefill/tests/test_graph_model.py#L195-L210).

### M18. The inherited gate compares each token's normalized key/query features with learned base keys.

- **Origin:** **Inherited behavior under the approved unchanged-gate decision.**
- **Rationale:** Each head produces token keys and query-group queries, applies
  RMS normalization, compares the matching token pair against `gate_sink`
  learned base keys, converts that competition to `[0,1]`, and averages query
  groups.
- **Alternatives:** A direct MLP probability, dot product without base keys, or
  a new graph-specific classifier.
- **Code:** [original full gate](../prefill/attention/gate.py#L47-L94),
  [headwise equivalent](../prefill/graph/model.py#L215-L246).

### M19. Only the hard FAISS builder is production-selectable in version 1.

- **Origin:** **Approved plan, Graph Construction and Parallelism** — “V1 ships
  the FAISS builder and a small test builder proving learnable edge weights
  receive gradients.”
- **Rationale:** The module seam proves learnable weighted topology is possible
  without adding a builder registry before a second production builder exists.
- **Alternatives:** Add a CLI builder registry now, ship a soft learned builder,
  or remove the general seam until it is needed.
- **Code:** [training construction](../prefill/train_graph.py#L701-L721),
  [learnable test builder](../prefill/tests/test_graph_model.py#L213-L240).

### M20. A fresh gate has 16 learned base keys per KV head, and graph training requires a positive sink count.

- **Origin:** **Implementation-only.**
- **Rationale:** Sixteen matches the released FastKVzip gate convention. A
  checkpoint overrides the default, but an explicit conflicting sink fails.
- **Alternatives:** Permit zero base keys where `Weight` supports it, infer the
  sink only from checkpoints, or expose model-specific defaults.
- **Code:** [default and validation](../prefill/train_graph.py#L257-L287),
  [base-key parameter](../prefill/attention/gate.py#L64-L75).

## 4. FAISS topology decisions

### F1. FAISS graph construction runs on CPU and loops through graphs sequentially.

- **Origin:** **Implementation-only.** You asked to “parallelize over layers and
  heads”; current graph microbatching parallelizes GIN/gate math, but not this
  FAISS loop.
- **Rationale:** This keeps installation on `faiss-cpu` and gives every graph an
  independent index.
- **Alternatives:** GPU FAISS, concurrent CPU indexes, Torch batched exact KNN,
  or one masked global index.
- **Code:** [per-graph CPU loop](../prefill/graph/builder.py#L126-L139),
  [dependency](../prefill/requirements.txt#L1-L3).

Graph microbatching batches `A`, `B`, and GIN. FAISS and the headwise gate
adapter still loop over the graphs in Python, so the requested layer/head
parallelism is only partial.

### F2. Neighbor similarity is raw maximum inner product on unnormalized `z`.

- **Origin:** **Approved plan, FAISS defaults** — “Maximum-inner-product
  search.”
- **Rationale:** Both vector direction and magnitude can influence topology.
- **Alternatives:** L2-normalized cosine similarity, Euclidean distance, or a
  learned similarity function.
- **Code:** [FAISS index metrics](../prefill/graph/builder.py#L64-L85).

FAISS distances are discarded. Hard FAISS edges therefore have no edge weights.

### F3. IVF-Flat is the default and IVF-PQ is configurable.

- **Origin:** **User formulation** — “FAISS IVF-Flat as default ... option to
  choose `IndexIVFPQ`.” The concrete defaults and automatic PQ divisor come
  from the **approved plan, FAISS defaults**.
- **Rationale:** IVF-Flat preserves full vectors; IVF-PQ trades recall for lower
  search cost and memory when `R` grows.
- **Alternatives:** Exact FlatIP, HNSW, LSH, or automatic index selection.
- **Code:** [builder modes](../prefill/graph/builder.py#L27-L50),
  [index construction](../prefill/graph/builder.py#L64-L89).

Defaults are `nlist=256`, `nprobe=16`, `pq_bits=8`. Automatic `pq_m` is the
largest divisor of `R` no larger than eight, as required by the plan.
See [`auto_pq_m`](../prefill/graph/builder.py#L52-L56) and its use in
[`_make_index`](../prefill/graph/builder.py#L75-L77).

### F4. Every graph construction creates, fills, and discards a fresh index; every IVF index is also retrained.

- **Origin:** **Implementation-only.**
- **Rationale:** Fresh construction avoids keeping neighbors or quantizers
  across updates to `A`. Exact Flat indexes need no training; IVF-Flat and
  IVF-PQ train from the current graph's vectors.
- **Alternatives:** Reuse a quantizer, rebuild every few updates, cache during a
  frozen phase, or use exact search without training.
- **Code:** [`_make_index`](../prefill/graph/builder.py#L64-L89),
  [per-graph call](../prefill/graph/builder.py#L126-L139).

In default two-phase training, graph construction and GIN propagation happen
once in the gate phase and again in the graph phase. There is still only one
graph optimizer update per context. Within one context the hard topology could
theoretically be cached; current code does not do so.

### F5. “Short context” has a concrete fallback threshold chosen by the implementation.

- **Origin:** **Approved-plan fallback; implementation-only threshold.** The plan
  requires “Exact `IndexFlatIP` fallback for short contexts.”
- **Rationale:** IVF-Flat falls back when `T<nlist`; IVF-PQ falls back when
  `T<max(nlist,2^pq_bits)`.
- **Alternatives:** Use FAISS's larger recommended training-sample heuristic,
  treat equality as short, or always use exact search below a latency threshold.
- **Code:** [`_uses_exact_search`](../prefill/graph/builder.py#L58-L68).

### F6. FAISS candidates are post-processed to guarantee exactly `min(k,T-1)` valid neighbors.

- **Origin:** **Implementation-only.**
- **Rationale:** Searching the indexed vectors against themselves returns self;
  IVF may also return `-1` or too few usable candidates.
- **Alternatives:** Accept variable degree, request a larger candidate pool, or
  use exact search whenever IVF is incomplete.
- **Code:** [`_neighbors`](../prefill/graph/builder.py#L91-L111).

The code removes self, negatives, and duplicates. Missing neighbors are filled
by a stable exact inner-product ordering.

### F7. Edges point `neighbor → target` and batched graphs are one disconnected index space.

- **Origin:** **Approved plan, FAISS defaults** — “Directed `neighbor → token`
  edges. No self-edges.”
- **Rationale:** GIN can gather `x[source]` and sum into each `target`; integer
  offsets prevent edges crossing graph boundaries.
- **Alternatives:** Reverse edges, symmetrize KNN, use mutual KNN, or represent a
  PyG batch vector instead of offsets.
- **Code:** [edge assembly](../prefill/graph/builder.py#L121-L145),
  [aggregation](../prefill/graph/model.py#L192-L207).

### F8. Hard neighbor choice is nondifferentiable, but live node features and optional edge weights remain differentiable.

- **Origin:** **User formulation for learnable builders** — “The current initial
  k-nn graph builder does not have learnable weights ... support graph builders
  that have learnable weights.” The no-caller-detach and FAISS-only-detach
  boundary comes from the **approved plan, Graph Construction and
  Parallelism**.
- **Rationale:** Only the CPU copy used by FAISS is detached. GIN still consumes
  live `z`, and a future builder's `edge_weight` enters message multiplication.
- **Alternatives:** Detach all graph inputs, use soft adjacency, straight-through
  neighbor selection, or a differentiable KNN relaxation.
- **Code:** [search detach](../prefill/graph/builder.py#L128-L135),
  [live GIN call](../prefill/graph/model.py#L364-L369).

### F9. `k` is capped for tiny contexts, and a one-token context has no edges.

- **Origin:** **Implementation-only.**
- **Rationale:** A token has at most `T-1` non-self neighbors; GIN's explicit
  self term still processes a singleton.
- **Alternatives:** Reject `k>=T`, add a self-edge, or pad with repeated
  neighbors.
- **Code:** [effective k and empty graph](../prefill/graph/builder.py#L113-L125).

### F10. IVF probing is clamped and FAISS state is not checkpointed.

- **Origin:** **Implementation-only.**
- **Rationale:** `nprobe=min(nprobe,nlist)` prevents an invalid setting; indexes
  are temporary products of current nodes.
- **Alternatives:** Reject oversized `nprobe`, checkpoint quantizers, expose a
  FAISS clustering seed, or cache topology.
- **Code:** [probe/train/add](../prefill/graph/builder.py#L86-L89),
  [saved architecture config](../prefill/train_graph.py#L552-L582).

### F11. KNN can connect any two context positions, including a later token into an earlier token.

- **Origin:** **Implementation-only consequence of whole-context KNN.**
- **Rationale:** Scoring happens after the complete causal prefill, so the graph
  is an offline importance model rather than part of autoregressive generation.
- **Alternatives:** Restrict sources to earlier positions, add relative-position
  features, or use local/causal KNN.
- **Code:** [all-node search](../prefill/graph/builder.py#L91-L110).

### F12. FAISS indexes are assembled with explicit constructors rather than `faiss.index_factory`.

- **Origin:** **Implementation-only.** You later raised the alternative:
  “Sounds like using faiss index factory is cleaner right?”
- **Rationale:** Explicit constructors make the Flat fallback, metric,
  quantizer, `nprobe`, and IVF-PQ arguments visible as Python values.
- **Alternatives:** Build `IVF...,Flat`/`IVF...,PQ...` descriptions with
  `faiss.index_factory`; this change would be localized mainly to
  `_make_index`.
- **Code:** [`_make_index`](../prefill/graph/builder.py#L64-L89).

## 5. Training and gradient decisions

### T1. Two-phase training is the default; joint training is the only public alternative.

- **Origin:** **User agreement** — “let's do this two phase training. But I want
  it as an optional flag. This should be the default ... regular training where
  both are training in the same phase.”
- **Rationale:** Two-phase preserves the original gate's many token updates;
  joint mode remains available for direct end-to-end learning.
- **Alternatives:** Joint-only training, graph-first alternation, or expose
  gate-only and graph-only modes publicly.
- **Code:** [CLI modes](../prefill/train_graph.py#L83-L88),
  [`train_context`](../prefill/graph/training.py#L788-L817).

`GraphTrainer` has internal gate-only and graph-only modes for composition and
tests, but the CLI rejects them.

### T2. The gate phase trains on current mixed features, not raw hidden states.

- **Origin:** **User formulation** — “the input to the gate in the gate phase is
  the mixed hidden states.”
- **Rationale:** The gate learns the representation it will receive at
  evaluation instead of locking onto a raw-only path.
- **Alternatives:** Pretrain on raw hidden states, gradually introduce `delta`,
  or alternate raw and mixed inputs.
- **Code:** [gate phase scoring](../prefill/graph/training.py#L584-L630),
  [mix operation](../prefill/graph/model.py#L218-L245).

With checkpoint-driven zero `B`, the first gate phase is raw FastKVzip by
design. Later contexts see the graph path updated by previous contexts.

### T3. Gate phase computes each complete graph once, stores `u` on CPU, and reuses it for all token updates.

- **Origin:** **Approved plan, Two-phase Gate phase** — “Compute current mixed
  graph outputs” and “Use 1,000-token batches.” Detaching and caching every `u`
  on CPU is **implementation-only**.
- **Rationale:** GIN has already mixed all `T` nodes, so tokenwise gate work can
  be sliced without changing the result.
- **Alternatives:** Keep `u` on GPU, recompute GIN per token batch, or use
  activation paging.
- **Code:** [graph precompute/cache](../prefill/graph/training.py#L577-L610).

The CPU cache still contains all `[L×H,T,R]` propagated activations for the
current context. It trades CPU RAM for graph recomputation.

### T4. Graph/joint processing keeps full-context `z/u` on GPU; gate token updates reuse CPU `u`.

- **Origin:** **Approved plan for whole-context graphs** — “A context with `T`
  tokens remains one graph example.” GPU/CPU staging is
  **implementation-only**.
- **Rationale:** Whole-context neighbor construction and GIN propagation need
  all nodes at once; tokenwise `A`, `B`, gate, and loss work can be staged.
- **Alternatives:** Partition sparse message passing, move GIN to CPU, use
  subgraph sampling, or keep full hidden states on GPU.
- **Code:** [full `z` construction](../prefill/graph/training.py#L514-L533),
  [full propagation](../prefill/graph/training.py#L661-L670).

Thus token microbatch size reduces temporary `[M,chunk,D]` work. Graph, joint,
validation, and the transient gate precomputation still materialize
`[M,T,R]` graph activations and the edge list. The repeated gate token updates
use detached `u` slices brought back from CPU.

### T5. Gate token positions are shuffled once per context and each chunk is one optimizer step.

- **Origin:** **Approved plan, Two-phase Gate phase** — “Shuffle token
  positions. Use 1,000-token batches. Take `ceil(T/1000)` gate optimizer steps.”
- **Rationale:** It preserves ordinary stochastic gate minibatches and the old
  number of gate updates.
- **Alternatives:** Contiguous positions, one accumulated context step, or
  length-adaptive batch sizes.
- **Code:** [token chunks](../prefill/graph/training.py#L555-L558),
  [gate step loop](../prefill/graph/training.py#L599-L631).

Every chunk loss is divided by its own token count. The smaller final chunk is
therefore a full optimizer step, not a proportionally smaller update. Longer
contexts also cause more gate updates than shorter contexts.

The reported gate BCE is an online average of chunks evaluated at successive
gate states. It is not a full-context BCE evaluated once at the final gate.

### T6. Graph phase uses a staged backward that is mathematically equivalent to retaining the full autograd graph.

- **Origin:** **Approved plan, Gradient clarification** — token slices
  “accumulate [their] part of `dLoss/du`” and then backward through GIN once.
- **Rationale:** It keeps one full `z/u` graph but avoids retaining full
  hidden-width gate and `A` activations.
- **Alternatives:** Ordinary end-to-end backward, PyTorch activation
  checkpointing, or approximate/truncated gradients.
- **Code:** [proxy and GIN backward](../prefill/graph/training.py#L641-L705),
  [`A` recomputation](../prefill/graph/training.py#L706-L720).

The sequence is: construct `z` without its `A` graph, make live `z`, run GIN,
collect token losses into `u_proxy.grad`, backward once through GIN, then
recompute token slices of `A` with the saved `dLoss/dz`.

### T7. Graph gradients accumulate over every graph microbatch before one graph update.

- **Origin:** **Approved plan, Graph phase** — “Take one graph optimizer step
  after the entire context.”
- **Rationale:** Changing graph microbatch size changes memory and runtime, not
  the mathematical context gradient.
- **Alternatives:** Step after each layer/head batch or scale learning rate with
  graph microbatch size.
- **Code:** [one zero/one step](../prefill/graph/training.py#L641-L724).

GIN receives one accumulated backward per graph microbatch. The optimizer is
still stepped only after all graph microbatches finish.

### T8. Default two-phase mode builds and propagates graphs once in each phase.

- **Origin:** **Implementation-only clarification.**
- **Rationale:** Gate phase stores detached CPU `u`; graph phase reruns
  propagation to obtain an autograd-connected execution for graph-parameter
  gradients. The intervening gate update changes the loss head, not `z` or the
  hard topology.
- **Alternatives:** Cache hard topology across phases, separate topology from
  propagation, or use joint mode.
- **Code:** [gate propagation](../prefill/graph/training.py#L584-L597),
  [graph propagation](../prefill/graph/training.py#L661-L670).

“One graph update per context” refers to the optimizer update, not one total
FAISS/GIN execution across both phases.

### T9. Two-phase graph training uses the gate after that context's gate updates.

- **Origin:** **User agreement** — “I agree with everything you said. So let's
  do this two phase training.” The approved plan fixes the order as gate phase,
  then graph phase.
- **Rationale:** The graph learns residuals for the gate state that will survive
  the context, not the gate state from before its minibatches.
- **Alternatives:** Graph-first order, simultaneous gradients, or delayed gate
  updates.
- **Code:** [phase order](../prefill/graph/training.py#L797-L805).

### T10. Joint mode accumulates one whole-context loss and steps both disjoint optimizers once.

- **Origin:** **Approved plan, Joint mode** — “Take one optimizer step per
  context.”
- **Rationale:** Gate and graph gradients see the same pre-update model and stay
  invariant to microbatch choices.
- **Alternatives:** One optimizer with parameter groups, gate minibatch updates
  inside joint mode, or alternating steps.
- **Code:** [joint staging and steps](../prefill/graph/training.py#L641-L733).

If the gate is frozen, both `joint` and `two_phase` effectively become one
graph-only update. The CLI does not require a separate graph-only mode.

### T11. Graph/joint loss and reported BCE are uniform means over all layer/head/token positions.

- **Origin:** **Implementation-only normalization.**
- **Rationale:** Every token score contributes equally, and graph/token
  microbatch sizes are mathematically invariant for the graph-phase loss and
  gradient, apart from floating-point accumulation order.
- **Alternatives:** Weight layers, heads, score classes, datasets, or token
  positions differently.
- **Code:** [loss denominator](../prefill/graph/training.py#L687-L697),
  [reported mean](../prefill/graph/training.py#L728-L733).

There is no delta penalty, class balancing, ranking term, or auxiliary graph
loss. Delta energy share is observation only. Gate-phase optimization is the
exception described in T5: each shuffled token chunk is a separate mean-loss
optimizer step, so changing its size changes the optimization trajectory.

### T12. Graph-batch identities stay as CPU Python tuples.

- **Origin:** **Implementation-only.**
- **Rationale:** Python control flow never calls `.item()` on CUDA tensors and
  therefore avoids accidental synchronization.
- **Alternatives:** Device index tensors, PyG batch vectors, or a vectorized map
  without Python identities.
- **Code:** [`GraphBatch`](../prefill/graph/model.py#L36-L80),
  [batch generation](../prefill/graph/model.py#L338-L354).

### T13. Validation computes the same streamed graph BCE without updates.

- **Origin:** **Implementation-only.**
- **Rationale:** It measures the deployed whole-context scorer with the same
  memory path as training.
- **Alternatives:** Cache validation labels, use a smaller validation graph, or
  evaluate downstream generation during training.
- **Code:** [`evaluate_context`](../prefill/graph/training.py#L736-L781).

It uses `torch.no_grad()` but does not call `scorer.eval()`. Current graph/gate
modules have no dropout, so this has no present numerical effect. A future
stochastic builder or module would need an explicit mode switch.

### T14. Mixed precision is manual and has no autocast, GradScaler, or gradient clipping.

- **Origin:** **Implementation-only.**
- **Rationale:** Explicit compute casts plus FP32 masters make the current small
  graph path predictable.
- **Alternatives:** AMP/autocast with scaling, full FP32 training, and/or norm
  clipping.
- **Code:** [loss promotion](../prefill/graph/training.py#L481-L553),
  [direct optimizer step](../prefill/graph/training.py#L560-L566).

### T15. Graph microbatch `auto` means one layer's KV heads, and explicit values must fit the model.

- **Origin:** **User formulation** — “`--graph-microbatch-size` should be auto
  by default ... otherwise should be between `[1, l*h]` and we should raise an
  error.”
- **Rationale:** The layer-major graph order makes `H` graphs one complete
  layer, while validation prevents empty or oversized batches before teacher
  generation.
- **Alternatives:** Choose a batch from measured free memory, default to one
  graph, or allow a larger value and clamp it.
- **Code:** [resolver](../prefill/graph/model.py#L82-L107),
  [post-model validation](../prefill/train_graph.py#L694-L700).

### T16. Graph phase freezes gate weights but keeps the derivative through the gate input.

- **Origin:** **Approved plan, Graph phase** — “Freeze gate parameters. Keep
  gradients through the gate's inputs.”
- **Rationale:** BCE can train `B`, GIN, `A`, and differentiable builder weights
  through the fixed gate without changing the gate during its graph update.
- **Alternatives:** Detach before the gate, train gate and graph together, or
  replace the gate with a separate graph-only loss head.
- **Code:** [gate freeze context](../prefill/graph/training.py#L641-L660),
  [loss backward through mixed input](../prefill/graph/training.py#L672-L704).

### T17. A graph microbatch materializes one hidden slice per graph, including repeated copies for heads in one layer.

- **Origin:** **Implementation-only.**
- **Rationale:** A uniform `[M,chunk,D]` tensor makes per-graph projection and
  head-specific scoring simple, even when several graph IDs share one layer's
  original hidden states.
- **Alternatives:** Broadcast one layer tensor into head-specific projections,
  group batches by layer without copies, or keep the current simpler staging.
- **Code:** [training hidden stack](../prefill/graph/training.py#L508-L512),
  [evaluation hidden stack](../prefill/graph/evaluation.py#L349-L361).

This duplication is token-microbatched, not a full `[H,T,D]` persistent GPU
cache. It still contributes to the temporary memory of each token slice.

## 6. Data order, optimization, checkpoints, and logging

### O1. Dataset membership and traversal order are fixed.

- **Origin:** **Approved plan, Training data** — the exact FineWeb indices are
  listed for training and validation.
- **Rationale:** It makes a small reproduction run deterministic and gives a
  simple next-item resume cursor.
- **Alternatives:** A manifest, shuffled contexts, length bucketing, or a
  configurable split.
- **Code:** [keys](../prefill/train_graph.py#L38-L45),
  [cursor traversal](../prefill/train_graph.py#L457-L506).

Training always visits 29 `fineweb_10k` contexts, then five
`fineweb_10k_cat` contexts. Context order is not shuffled between epochs.

### O2. One epoch is the default; validation follows all 34 training contexts.

- **Origin:** **Approved plan, Initialization and Checkpoints** — “Default
  epochs: `1`.” Placing validation only after the full training split is
  **implementation-only**.
- **Rationale:** It matches the small teacher-data reproduction target and keeps
  validation/checkpoint semantics simple.
- **Alternatives:** Mid-epoch validation, early stopping, or a fixed number of
  context updates.
- **Code:** [epoch default](../prefill/train_graph.py#L52-L58),
  [epoch loop](../prefill/train_graph.py#L866-L897),
  [cursor transitions](../prefill/train_graph.py#L474-L506).

Validation contains four contexts and generates their teacher data online
again. There is no validation cache.

### O3. Validation selection uses the unweighted mean of four per-context BCE values.

- **Origin:** **Implementation-only.**
- **Rationale:** Every validation context has equal influence regardless of its
  length.
- **Alternatives:** Token-weighted mean, separate per-dataset means, or a
  downstream task metric.
- **Code:** [validation accumulation](../prefill/train_graph.py#L474-L506).

### O4. Gate and graph parameters use separate AdamW optimizers.

- **Origin:** **Approved plan, Optimizers and Schedulers** — “Use AdamW,” with
  separate gate and graph learning rates and scheduler specifications.
- **Rationale:** Two-phase mode needs different step frequencies and default
  learning rates. The graph optimizer includes `A`, GIN, `B`, and builder
  parameters.
- **Alternatives:** One AdamW with parameter groups, SGD, Adafactor, or fused
  optimizers.
- **Code:** [`build_adamw_optimizers`](../prefill/graph/training.py#L237-L275).

Default weight decay is `0.01` and applies uniformly to weights, biases, norms,
GIN epsilon, and gate base parameters. AdamW betas/epsilon use PyTorch defaults.
Those details were not in the plan.

### O5. Two-phase learning rates default to `1e-4` for gate and `1e-3` for graph; joint defaults both to `1e-4`.

- **Origin:** **Approved plan, Optimizers and Schedulers** — “Gate learning
  rate: `1e-4`; Graph learning rate: `1e-3`” and in joint mode “Both learning
  rates: `1e-4`.”
- **Rationale:** The graph path starts faster in staged training, while joint
  mode keeps symmetric optimizer settings.
- **Alternatives:** Tune per model, use one shared optimizer, or scale with
  effective token/context batch size.
- **Code:** [resolution](../prefill/train_graph.py#L318-L361).

In joint mode a missing peer setting is copied. Active gate/graph learning rates
and scheduler specifications must be exactly equal. Equality is waived when the
gate is frozen.

### O6. Schedulers are resolved from PyTorch classes using JSON-only constructor arguments.

- **Origin:** **User formulation for scheduler support** — “support learning
  rate schedulers supported by pytorch for both phases.” Name resolution and
  JSON constructor arguments come from the **approved plan, Optimizers and
  Schedulers**.
- **Rationale:** A class and JSON object can be validated before the large LLM is
  loaded and serialized safely in config.
- **Alternatives:** Curated scheduler choices, Python configuration files, or
  callable schedules.
- **Code:** [scheduler parsing](../prefill/graph/training.py#L76-L125),
  [CLI resolution](../prefill/train_graph.py#L208-L219).

Schedulers requiring Python callables, such as ordinary `LambdaLR` usage,
cannot be expressed. Normal schedulers step after each optimizer step. Therefore
a two-phase gate scheduler steps `ceil(T/1000)` times per context; the graph
scheduler steps once. `ReduceLROnPlateau` waits for the completed validation
mean.

### O7. Released, local legacy, and graph-training gate checkpoints are accepted.

- **Origin:** **Approved plan, Initialization and Checkpoints** —
  “`--gate-checkpoint` accepts `fastkvzip` or a local checkpoint.” Supporting
  both legacy module lists and full graph payloads is **implementation-only**.
- **Rationale:** Experiments can start from released weights, old module-list
  files, or another graph run.
- **Alternatives:** One versioned gate-only format or always require full graph
  checkpoints.
- **Code:** [gate metadata/loading](../prefill/train_graph.py#L159-L195),
  [released-gate resolution](../prefill/train_graph.py#L539-L548),
  [`load_gate_checkpoint`](../prefill/graph/training.py#L411-L420).

Explicit gate-dimension or sink conflicts fail instead of silently reshaping
weights. Compute dtype is inferred from checkpoint metadata/state and enforced
when a graph checkpoint is loaded; there is no compute-dtype CLI override.
Freezing a gate without checkpoint/resume is invalid.

### O8. Resume compatibility checks only the normalized graph-training config.

- **Origin:** **Implementation-only.**
- **Rationale:** Architecture and optimizer/scheduler identity are locked, while
  invocation controls such as a pilot limit can change.
- **Alternatives:** Save and classify every field as immutable or mutable, or
  offer separate strict-resume and fine-tune modes.
- **Code:** [normalized fields](../prefill/train_graph.py#L552-L590),
  [option merging](../prefill/train_graph.py#L198-L416).

The normalized config omits weight decay, epoch target, seed, `max_contexts`,
output directory, W&B settings, dataset revision/content hashes, library
versions, tokenizer revision, and Git commit. Loaded optimizer state silently
restores the old weight decay even if a different resume CLI value was given.
Epochs may intentionally be extended, and `max_contexts` intentionally resets
per invocation.

### O9. `last.pt` points to the next context and is atomically replaced after every successfully processed context.

- **Origin:** **Approved plan, Initialization and Checkpoints** — “Save
  `best.pt` and `last.pt` at context boundaries.” Next-item cursor semantics and
  atomic replacement are **implementation-only**.
- **Rationale:** A resumed process does not repeat a successfully checkpointed
  update, and a failed write preserves the previous checkpoint.
- **Alternatives:** Store last-completed cursor, version every step, checkpoint
  once per epoch, or use a journal.
- **Code:** [cursor/save order](../prefill/train_graph.py#L866-L896),
  [atomic write](../prefill/graph/training.py#L353-L363).

There is no mid-context checkpoint. Fixed `best.pt` and `last.pt` overwrite
history. The atomic replace does not explicitly `fsync` file or directory.

The custom W&B log happens before cursor advancement and checkpoint saving. A
crash or save failure in between can make a resumed run attempt and log that
same context/step again; logging and checkpointing are not one transaction.

### O10. A checkpoint contains same-environment context-boundary continuation state but no teacher example.

- **Origin:** **Approved plan, Checkpoints** — graph/gate weights, optimizer and
  scheduler states, configuration, model ID, prefix, prefill size, cursor, RNG,
  and W&B run ID are required.
- **Rationale:** Student, optimizer, scheduler, cursor, and tracked RNG state can
  continue at a context boundary under the same code, data, tokenizer, and
  environment without storing activations or labels.
- **Alternatives:** Separate inference and trainer checkpoints, include the
  current teacher example, or save immutable step snapshots.
- **Code:** [payload](../prefill/graph/training.py#L299-L352),
  [restore](../prefill/graph/training.py#L366-L408).

Python, NumPy, CPU Torch, and available CUDA RNG states are captured and
restored when the corresponding runtime is available. FAISS state, postfix
tokens, dependency versions, and dataset state outside the cursor are not
captured.

### O11. Checkpoints are trusted pickle payloads.

- **Origin:** **Implementation-only.**
- **Rationale:** Optimizer state and Python/NumPy RNG objects fit naturally in a
  normal `torch.save` payload.
- **Alternatives:** A weights-only tensor file plus JSON metadata and separately
  encoded RNG state.
- **Code:** [training load](../prefill/train_graph.py#L222-L225),
  [checkpoint load](../prefill/graph/training.py#L373-L408).

`weights_only=False` means a checkpoint must come from a trusted source.

### O12. The trainer makes one custom W&B log call per context attempt.

- **Origin:** **User formulation for the metric set** — “Let's report: gate and
  graph BCE, bounded delta energy share, forward time, backward time, gpu memory
  and utilization, learning rate.” Logging once per context comes from the
  **approved plan, W&B Logging**.
- **Rationale:** It avoids per-token logging overhead and parameter/gradient
  hooks.
- **Alternatives:** Optimizer-step logs, epoch-only summaries, or richer
  profiling runs.
- **Code:** [`run_and_log_context`](../prefill/train_graph.py#L627-L683).

The custom keys are mode-dependent, and disabled W&B emits nothing externally.
GPU utilization is not a custom context key; it is left to W&B's asynchronous
system monitor. No `wandb.watch`, parameter histograms, or gradient histograms
are enabled.

Ordinary scheduler steps happen before this log, so its learning rate is the
post-step value for the next update. `ReduceLROnPlateau` steps only after the
completed validation pass, after its final context has already been logged.

### O13. GPU peaks include teacher generation, but phase timings include only student regions.

- **Origin:** **Implementation-only interpretation of the logging plan.**
- **Rationale:** One peak answers “can this context fit?”, while CUDA events
  bracket gate/graph forward and backward regions.
- **Alternatives:** Separate teacher/student peaks and add teacher, optimizer,
  checkpoint, and total-context wall times.
- **Code:** [peak reset before teacher](../prefill/train_graph.py#L866-L884),
  [timing collector](../prefill/graph/training.py#L193-L234),
  [gate regions](../prefill/graph/training.py#L591-L630),
  [graph regions](../prefill/graph/training.py#L664-L720).

CUDA events are resolved with one synchronization at the context boundary.
Optimizer/scheduler steps, teacher generation, W&B, and checkpoint writing are
outside the forward/backward timers. The timed forward regions do include
hidden transfers and synchronous CPU FAISS work; they are not pure GPU-kernel
breakdowns.

### O14. Training and validation graph BCE share the same W&B key.

- **Origin:** **Implementation-only.**
- **Rationale:** The logger keeps the exact small metric set requested.
- **Alternatives:** Add a `phase` field, use `train/graph_bce` and
  `validation/graph_bce`, and log completed validation mean/best separately.
- **Code:** [validation/result mapping](../prefill/train_graph.py#L647-L678).

The completed validation mean and best value are used for scheduling and
checkpoint selection but are not separately logged.

### O15. `--max-contexts` is an invocation-local pilot limit that counts both training and validation contexts.

- **Origin:** **Implementation-only, added to support the approved one-context
  Slurm pilot.**
- **Rationale:** A pilot can stop safely after `last.pt`, then resume from the
  next cursor without making the limit persistent.
- **Alternatives:** A train-only limit, a permanent global budget, or a separate
  pilot script.
- **Code:** [loop condition](../prefill/train_graph.py#L866-L897),
  [README pilot](../prefill/README.md#L59-L76).

### O16. Cleanup relies on reference deletion and allocator reuse.

- **Origin:** **Implementation-only.**
- **Rationale:** PyTorch can reuse reserved CUDA blocks without costly allocator
  flushes.
- **Alternatives:** `gc.collect`, `torch.cuda.empty_cache`, explicit cache
  context managers, or memory-pool controls.
- **Code:** [payload/KV/example deletion](../prefill/train_graph.py#L778-L885).

### O17. Datasets and wrappers are loaded lazily once per dataset name and kept for the run.

- **Origin:** **Implementation-only.**
- **Rationale:** The two small FineWeb sources are not repeatedly downloaded,
  parsed, or wrapped for every context.
- **Alternatives:** Load one context at a time, eagerly load both datasets at
  startup, or stream from an explicit manifest.
- **Code:** [wrapper cache](../prefill/train_graph.py#L825-L847).

The cached dataset objects add CPU lifetime beyond one `TeacherExample`. Before
each prefill, the saved training prefix is reinstalled because constructing a
wrapper mutates the model template.

### O18. Validation and initialization are ordered to fail inexpensive conditions early.

- **Origin:** **Implementation-only.**
- **Rationale:** Model-independent CLI/scheduler/checkpoint checks happen before
  W&B and the LLM; W&B login/init happens before model loading; the
  model-dependent graph-microbatch range is checked before datasets and teacher
  generation.
- **Alternatives:** Initialize W&B last, read model dimensions from config
  without weights, or allow failures only when an option is first used.
- **Code:** [runner setup order](../prefill/train_graph.py#L770-L805).

### O19. Fresh runs seed common RNGs with zero but do not request deterministic algorithms.

- **Origin:** **Implementation-only.**
- **Rationale:** Python, NumPy, Torch CPU, and CUDA begin from a repeatable seed,
  while kernels are free to use their normal performant implementations.
- **Alternatives:** Enable deterministic algorithms, use named generators for
  token shuffling, or expose/capture FAISS and dataset-specific RNG state.
- **Code:** [seed default](../prefill/train_graph.py#L52-L59),
  [seed application](../prefill/train_graph.py#L686-L691).

On resume, model/optimizer/scheduler objects are constructed first and then the
saved RNG state is restored. External library and dataset revisions remain
outside this guarantee.

### O20. Resume continues the stored W&B run when possible and always finishes an initialized run.

- **Origin:** **Approved plan, Checkpoints** — checkpoints include the “W&B run
  ID.” The exact `resume="allow"` lifecycle is **implementation-only**.
- **Rationale:** Context logs can continue at the stored step while failures
  after initialization still close the client cleanly.
- **Alternatives:** Start a child run for every resume, require the old run to
  exist with `resume="must"`, or decouple training checkpoints from W&B state.
- **Code:** [W&B initialization](../prefill/train_graph.py#L593-L606),
  [run ID/step restoration](../prefill/train_graph.py#L784-L823),
  [final cleanup](../prefill/train_graph.py#L897-L899).

## 7. Evaluation and pruning decisions

### E1. Graph evaluation has its own entry point and reuses the old benchmark helpers.

- **Origin:** **User formulation** — “What if we have a brand new script? Can
  we reuse relevant methods and keep it relatively slim?”
- **Rationale:** `eval.py` stays unchanged, while dataset grouping, retention
  ratios, generation, evaluation, timing, and result formats remain comparable
  with the baseline.
- **Alternatives:** Modify `eval.py`, duplicate its helpers, or extract a shared
  runner used by both scripts.
- **Code:** [`main`](../prefill/eval_graph.py#L119-L124),
  [reused imports](../prefill/eval_graph.py#L7-L20).

The call chain is:

```text
python eval_graph.py
  -> main()
  -> run_evaluation()
  -> load_evaluation_checkpoint()
  -> build_evaluation_runtime()
  -> DataWrapper.prefill_context()
  -> score_context_cache()
  -> DataWrapper.generate_answer()
  -> RetainCache.prune() + Evaluator()
  -> save_result()
```

### E2. Checkpoint metadata and reconstruction-critical tensors are validated on CPU before the LLM is loaded.

- **Origin:** **Implementation-only.**
- **Rationale:** Bad format, dimensions, dtypes, prefix IDs, and key
  projection/gate shapes fail before an expensive model allocation.
- **Alternatives:** Trust `load_state_dict`, validate after model construction,
  or save a small metadata file beside the weights.
- **Code:** [`_validate_checkpoint`](../prefill/graph/evaluation.py#L73-L206),
  [`load_evaluation_checkpoint`](../prefill/graph/evaluation.py#L209-L218),
  [runner order](../prefill/eval_graph.py#L60-L63).

The accepted format is exactly version `1`. Extra training payload/config
fields are tolerated. Missing or unexpected scorer keys and other shape
mismatches are rejected only by the later strict load after reconstruction.
Loading uses `weights_only=False`, so the file must be trusted.

### E3. The checkpoint model ID is authoritative, and `--model` is only an exact-match assertion.

- **Origin:** **Implementation-only.**
- **Rationale:** A CLI value cannot silently pair graph weights with another
  language model.
- **Alternatives:** Remove `--model`, permit model aliases/revisions, or allow an
  unsafe override after structural checks.
- **Code:** [override check](../prefill/graph/evaluation.py#L209-L218),
  [runtime model construction](../prefill/graph/evaluation.py#L292-L304).

The loaded model's layer count, KV-head count, query-group count, and hidden
width must also exactly match the checkpoint. Both ordinary configs and nested
`text_config` are supported.

### E4. Evaluation uses a full retain cache, an ungated LLM, and a separate graph scorer.

- **Origin:** **Approved plan, Standalone Evaluation** — “Load `ModelKVzip`
  with no built-in gate” and “Load the graph checkpoint.”
- **Rationale:** Graph scores are computed once after prefill, while the old
  cache/pruning code can evaluate several ratios without another prefill.
- **Alternatives:** Insert graph-aware gates into attention, physically evict
  KV tensors, or prefill separately for every ratio.
- **Code:** [`build_evaluation_runtime`](../prefill/graph/evaluation.py#L292-L304),
  [`RetainCache`](../prefill/attention/kvcache.py#L234-L260).

“Pruning” in this evaluator is masking. The complete K/V tensors stay in full on
the model device, so this measures output quality at a ratio, not the memory
savings of physical eviction.

### E5. Evaluation reconstructs architecture and microbatch settings from the checkpoint without CLI overrides.

- **Origin:** **Implementation-only.** The approved plan requires loading the
  graph checkpoint, but the exact structural replay and lack of runtime
  microbatch overrides were chosen in code.
- **Rationale:** Evaluation cannot accidentally change graph meaning or the
  tested memory-staging path.
- **Alternatives:** Expose runtime-only token/graph microbatch overrides, or
  re-resolve `auto` for the evaluation GPU.
- **Code:** [saved evaluation fields](../prefill/graph/evaluation.py#L20-L56),
  [runtime reconstruction](../prefill/graph/evaluation.py#L235-L289),
  [CLI](../prefill/eval_graph.py#L23-L39),
  [saved microbatch consumption](../prefill/eval_graph.py#L92-L99).

The replayed fields are gate dimension/sink, compute dtype, model dimensions,
graph dimension/depth/neighbors, FAISS settings, and concrete graph/token
microbatches. Training mode, optimizers, and schedulers are not replayed. The
saved graph microbatch is already an integer; evaluation does not recompute
`auto`, even on different hardware. The scorer is put in evaluation mode, and
the scoring entry points run under `torch.inference_mode()`.

### E6. `DataWrapper` installs the current dataset template, then only the saved training prefix is restored.

- **Origin:** **User formulation for the prefix** — “the hidden states for the
  context are generated exactly the same in training and evaluation.” The
  postfix behavior is **implementation-only**.
- **Rationale:** Context states see the exact trained prefix, while the current
  dataset's query/assistant postfix is preserved.
- **Alternatives:** Checkpoint and restore both prefix and postfix, serialize the
  full chat template, or reject a postfix mismatch.
- **Code:** [template mutation](../prefill/data/wrapper.py#L27-L32),
  [restore order](../prefill/eval_graph.py#L74-L80),
  [prefix restore](../prefill/graph/evaluation.py#L502-L505).

This reproduces the saved tokens before the context and the context's prefill
boundaries. It does not snapshot the post-context prompt, tokenizer/model
revision, or software environment.

### E7. Evaluation prefills with the saved chunk size, captures hidden states, and disables legacy scoring.

- **Origin:** **Approved plan, Standalone Evaluation** — “Prefill with
  `save_hidden=True` and `do_score=False`.”
- **Rationale:** It reproduces training context activations while preventing a
  released gate or reconstruction scorer from overwriting graph scores.
- **Alternatives:** Capture hidden states in a second forward, score during each
  attention call, or use a different evaluation chunk size.
- **Code:** [prefill call](../prefill/eval_graph.py#L85-L99),
  [chunked model prefill](../prefill/model/wrapper.py#L155-L227).

Every layer's prefill chunks are concatenated as CPU hidden tensors. The K/V
cache remains on the model device.

### E8. Hidden-cache validation is deliberately strict.

- **Origin:** **Implementation-only.**
- **Rationale:** Exactly `L` CPU tensors shaped `[1,P+T,D]` prevent silent layer
  loss, query contamination, device surprises, and score misalignment.
- **Alternatives:** Accept GPU tensors, allow longer tensors and slice them, or
  add adapters for hybrid/static cache layouts.
- **Code:** [`_validate_hidden_cache`](../prefill/graph/evaluation.py#L307-L346).

The context interval must be nonempty and end exactly at the captured length.
This is why the current path rejects hybrid/static models.

### E9. Evaluation transfers hidden states twice per graph batch but propagates one complete context graph.

- **Origin:** **Approved-plan whole-context graph; implementation-only two-pass
  staging.**
- **Rationale:** The first token-chunk pass fills complete `z`, GIN produces
  complete `u`, and the second pass retrieves `x` again for `B` and the gate.
  Full `[L,T,D]` hidden states never need to reside on GPU together.
- **Alternatives:** Keep hidden states on GPU for one pass, recompute `A`, run
  graph propagation on CPU, or partition the sparse graph.
- **Code:** [`score_hidden_cache`](../prefill/graph/evaluation.py#L364-L434).

Full `[M,T,R]` `z` and `u` still live on GPU for each graph microbatch. Token
microbatching only bounds the hidden-width transfers and tokenwise work.

### E10. `score_context_cache` assigns `[L,1,H,T]` and always clears the hidden cache it receives.

- **Origin:** **Approved plan, Standalone Evaluation** — “Assign scores with
  shape `[layers,1,heads,T]`” and “Clear hidden states.”
- **Rationale:** The shape exactly matches `KVScore.threshold`, and `finally`
  cleanup releases CPU activations even when scoring fails.
- **Alternatives:** Adapt pruning to `[L,H,T]`, retain hidden states for
  inspection, or return scores without mutating the cache.
- **Code:** [reshape](../prefill/graph/evaluation.py#L431-L434),
  [assignment and cleanup](../prefill/graph/evaluation.py#L460-L499).

The scorer device and cache device must match. The implementation does not
perform an implicit score transfer. A failure during prefill occurs before this
`finally` and is outside its cleanup guarantee.

### E11. The local suffix is favored by replacing its scores with the global maximum.

- **Origin:** **Approved plan, Standalone Evaluation** — “Apply the existing
  protected local window.” The exact score replacement is **inherited
  behavior**.
- **Rationale:** It preserves the baseline recency preference without changing
  the thresholding interface.
- **Alternatives:** Use an explicit immutable mask, assign infinity/dtype max,
  or learn recency inside the graph.
- **Code:** [`protect_local_window`](../prefill/graph/evaluation.py#L437-L449),
  [baseline rule](../prefill/model/wrapper.py#L183-L225).

For `T < prefill_chunk`, the window is `floor(0.02×T)`; otherwise it is the CLI
window. This is not an absolute guarantee: strict thresholds, tied maximum
scores, or a budget below the window can still exclude those positions.

`prefill_chunk` is restored from the checkpoint. The long-context window is a
runtime CLI value, defaults to `4096`, and is not checkpointed. A window larger
than `T` covers the whole context through ordinary slice semantics.

### E12. Evaluation inherits ratios `[0.75,0.5,0.4,0.3,0.2]` and four allocation levels.

- **Origin:** **Inherited behavior under the approved helper-reuse decision.**
- **Rationale:** Graph and baseline results use the same compression sweep and
  budget-allocation semantics.
- **Alternatives:** Make ratios part of the CLI, save them in the checkpoint, or
  define a single exact-top-k policy.
- **Code:** [ratios](../prefill/eval.py#L7-L9),
  [level CLI](../prefill/eval_graph.py#L30-L38),
  [threshold dispatch](../prefill/attention/score.py#L95-L170).

The default level is global `pair`. `pair-head` uses per-head top-k;
`pair-layer` thresholds each layer; `adakv-layer` first safeguards 20% of the
requested per-head budget. The baseline's gate-name-based automatic level
selection is not used; those two choices are **implementation-only**. Head and
layer modes report threshold `0`; only global `pair` reports its scalar
threshold.

### E13. Legacy global/layer thresholding is off by one, and ties can reduce retention further.

- **Origin:** **Inherited behavior.**
- **Rationale:** Preserving the old strict `score > threshold` rule keeps direct
  comparability with released results.
- **Alternatives:** Use `>=`, construct exact top-k masks, or add deterministic
  tie-breaking noise.
- **Code:** [global and layer thresholds](../prefill/attention/score.py#L110-L151),
  [reported true ratio](../prefill/attention/kvcache.py#L304-L314).

For global `pair`, `N` is every candidate score. For a layer mode, `N` is every
candidate within that layer. Both select threshold index
`max(floor(N×ratio)-1,0)` and then reject the threshold itself with strict `>`.
With distinct scores this keeps one fewer than the nominal count; ties can keep
even fewer. `pair-head` instead uses exact per-head top-k. The measured retained
fraction is saved beside the request.

### E14. Prefix, query, postfix, and generated positions are retained structurally, not scored by the graph.

- **Origin:** **User formulation** — “evicting only kv cache values of the
  context token positions.”
- **Rationale:** The graph score covers exactly `T` context positions. The
  retain-cache mask pads `True` on the left for the prefix and on the right for
  every later token.
- **Alternatives:** Assign sentinel scores to all protected regions or maintain
  a separate explicit immutable-position mask.
- **Code:** [context eviction range](../prefill/model/wrapper.py#L170-L175),
  [mask padding](../prefill/attention/kvcache.py#L329-L335).

Start/end-of-turn tokens are protected according to where the current template
places them: before the context they are in the prefix; after it they are in the
postfix/query side.

### E15. The full-cache answer is generated first, then every pruned ratio is regenerated from the same context cache.

- **Origin:** **Inherited behavior under the approved helper-reuse decision.**
- **Rationale:** Each pruned answer can be compared with the model's own
  uncompressed answer and ground truth without repeating prefill or graph
  scoring.
- **Alternatives:** Compare only to ground truth, re-prefill each ratio, or
  evaluate answer probabilities instead of generated text.
- **Code:** [full answer preparation](../prefill/data/wrapper.py#L66-L122),
  [ratio loop](../prefill/eval_graph.py#L100-L113),
  [comparison format](../prefill/utils/tester.py#L31-L44),
  [generation cache reset](../prefill/model/wrapper.py#L277-L309).

After every query generation, `RetainCache.slice()` removes query/generated KV
and returns the cache to its pre-query length. Ratios are therefore evaluated
from the same base cache. The mask changes; the full K/V tensors do not.

### E16. Evaluation uses greedy generation and dataset-specific maximum output lengths.

- **Origin:** **Inherited behavior.**
- **Rationale:** Deterministic decoding reduces noise when comparing retention
  ratios, while task-specific limits avoid excessive generation.
- **Alternatives:** Sampling with fixed RNG, beam search, or one universal
  length limit.
- **Code:** [generation defaults](../prefill/model/wrapper.py#L61-L83),
  [length selection](../prefill/utils/func.py#L27-L42).

The graph script calls `generate_answer(..., prob=False)`. It records generated
text comparisons, not the optional probability-difference metrics.

### E17. Dataset expansion and result layout remain baseline-compatible, with graph-specific tags.

- **Origin:** **Approved plan, Standalone Evaluation** for helper reuse; the
  graph tag is **implementation-only**.
- **Rationale:** Existing short/mid/long groups and Qwen/Gemma substitutions
  remain comparable, while `_graph` prevents collision with ordinary baseline
  folders.
- **Alternatives:** Store results under a separate graph root, include a
  checkpoint hash in every path, or use an immutable run manifest.
- **Code:** [dataset expansion](../prefill/eval.py#L12-L58),
  [tag normalization](../prefill/eval_graph.py#L42-L46),
  [result payload](../prefill/eval_graph.py#L103-L113),
  [result path](../prefill/utils/func.py#L45-L52).

The path is `results/<dataset>/<index>_<model><tag>/output-<level>.json`.
Running the same sample, tag, and level again overwrites that JSON file.
Each evaluator value is stored beside
`[requested_ratio, round(actual_ratio,4), round(threshold,4)]`.

### E18. Any structurally valid graph checkpoint can be evaluated.

- **Origin:** **Implementation-only.**
- **Rationale:** Both `last.pt` and `best.pt` are useful for pilots and debugging;
  inference does not need optimizers, schedulers, cursor, W&B, or RNG state.
- **Alternatives:** Require a completed-validation marker, accept only
  `best.pt`, or emit a warning for partial training.
- **Code:** [checkpoint validation](../prefill/graph/evaluation.py#L73-L218),
  [inference-only state restoration](../prefill/graph/evaluation.py#L288-L289),
  [optional training-state restoration](../prefill/graph/training.py#L373-L408).

The loader restores graph and gate weights strictly with `restore_rng=False`.
It does not decide whether those weights are scientifically ready.

### E19. The printed evaluation time covers the whole sample, not graph scoring alone.

- **Origin:** **Inherited behavior.**
- **Rationale:** One number captures prefill, graph scoring, full-answer
  generation, all five ratio passes and every generation within them,
  evaluation, and saving.
- **Alternatives:** Separate timers for prefill, graph construction, scoring,
  and each generation ratio.
- **Code:** [timestamp scope](../prefill/eval_graph.py#L82-L116),
  [`TimeStamp`](../prefill/utils/func.py#L74-L102).

### E20. Expensive evaluation collaborators are injectable for tests.

- **Origin:** **Implementation-only.**
- **Rationale:** The whole orchestration path can be tested on CPU without
  downloading a model, dataset, or writing real benchmark results.
- **Alternatives:** Monkeypatch module globals, test only small helpers, or run
  every test with the real stack.
- **Code:** [`run_evaluation` parameters](../prefill/eval_graph.py#L49-L69),
  [orchestration test](../prefill/tests/test_graph_eval.py#L475-L570).

### E21. Evaluation uses permissive sample ranges and aborts on the first uncaught sample error.

- **Origin:** **Implementation-only.**
- **Rationale:** Python range clipping keeps ordinary subset runs simple, while
  exceptions remain visible instead of silently skipping bad samples.
- **Alternatives:** Validate every range before model loading, isolate failures
  per sample, or save an explicit failure record and continue.
- **Code:** [range handling](../prefill/eval_graph.py#L60-L85),
  [sample lifecycle](../prefill/eval_graph.py#L85-L116),
  [window validation](../prefill/graph/evaluation.py#L437-L449).

Only negative `idx` or `num` is rejected, and that check happens after model
loading. `num=0` or an index beyond the dataset gives an empty run; the upper
bound is clipped to dataset length. A negative window fails only after that
sample's prefill. `score_context_cache` clears hidden state on scoring errors,
but there is no per-sample exception isolation and the explicit final `del`
runs only on success. JSON files from earlier samples remain on disk.

## 8. Object lifetime, device, and gradient map

This table connects the decisions above into one memory model.

| Object | Normal location | Lifetime | Receives task gradients? |
|---|---|---|---|
| Teacher LLM weights | GPU | Entire training process | No; teacher runs in inference mode. |
| Teacher K/V cache | GPU | One context's label generation | No; deleted before student work. |
| Captured layer inputs | CPU | One context | No; cloned into `TeacherExample`. |
| Teacher scores | CPU after extraction | One context | No; BCE targets. |
| Context IDs and metadata | CPU | One context | No; currently informational only. |
| Student gate parameters | GPU, FP32 masters for low-precision compute | Entire run | Gate phase and joint mode, unless frozen. |
| `A`, GIN, and `B` parameters | GPU, FP32 masters for low-precision compute | Entire run | Graph phase and joint mode. |
| FAISS index | CPU | One layer/head graph construction | No; freshly built and discarded. |
| `EdgeIndex` | Student device | One graph microbatch | No for hard FAISS endpoints. |
| Optional `edge_weight` | Student device | One graph microbatch | Yes if a builder produces it differentiably. |
| Full `z` and `u` | Student device | One graph microbatch | Yes in graph/joint; transient and gradient-free during gate precompute. |
| Gate-phase `u` cache | CPU | One gate phase | No; detached before gate token steps. |
| Evaluation hidden cache | CPU | One context until scoring returns/fails | No; cleared by `score_context_cache` once prefill has returned it. |
| Evaluation K/V cache | Model device | One sample across every ratio | No; retained in full and masked. |

The optimizer/update schedule is:

| Mode | Gate optimizer | Graph optimizer | Graph executions |
|---|---|---|---|
| Default two-phase | `ceil(T/1000)` steps | One step per context | One traversal of all graph microbatches for gate features, then one for graph gradients. |
| Joint | One step per context | One step per context | One traversal of all graph microbatches for the shared context loss. |
| Frozen gate | No gate optimizer | One step per context | One graph-phase traversal of all graph microbatches. |

## 9. Verification contract and current evidence

The approved plan explicitly requested the following checks. Each implemented
check is linked to the test that owns it.

Fresh audit result on 2026-08-27, run from `prefill/`:

```text
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python -m pytest -q tests
157 passed in 5.36s
```

| Required property | Current evidence |
|---|---|
| Headwise adapter equals the old full gate | [`test_graph_model.py`](../prefill/tests/test_graph_model.py#L242-L285) |
| Zero `B` plus a gate checkpoint reproduces old scores | [`test_graph_model.py`](../prefill/tests/test_graph_model.py#L287-L316) and [`test_graph_model.py`](../prefill/tests/test_graph_model.py#L480-L499) |
| Grouped GIN equals independent graphs | [`test_graph_model.py`](../prefill/tests/test_graph_model.py#L70-L193) |
| Graph/joint token chunk sizes preserve full-context loss and gradients | [`test_graph_training.py`](../prefill/tests/test_graph_training.py#L289-L393) |
| Graph microbatch sizes preserve loss and gradients | [`test_graph_training.py`](../prefill/tests/test_graph_training.py#L368-L393) and [`test_graph_model.py`](../prefill/tests/test_graph_model.py#L501-L517) |
| Staged backward equals ordinary float64 backward | [`test_graph_training.py`](../prefill/tests/test_graph_training.py#L289-L366) |
| Gate and graph optimizer step counts are exact | [`test_graph_training.py`](../prefill/tests/test_graph_training.py#L395-L440) and [`test_graph_training.py`](../prefill/tests/test_graph_training.py#L760-L807) |
| FAISS excludes self and supplies expected degree | [`test_graph_builder.py`](../prefill/tests/test_graph_builder.py#L15-L53) |
| Learnable edge weights receive gradients | [`test_graph_model.py`](../prefill/tests/test_graph_model.py#L213-L240) and [`test_graph_model.py`](../prefill/tests/test_graph_model.py#L382-L419) |
| Invalid graph microbatch values fail early | [`test_graph_model.py`](../prefill/tests/test_graph_model.py#L421-L445) and [`test_graph_train_cli.py`](../prefill/tests/test_graph_train_cli.py#L399-L430) |
| Joint mode rejects unequal LR/scheduler settings | [`test_graph_training.py`](../prefill/tests/test_graph_training.py#L164-L190) |
| Scheduler state survives checkpoint/resume | [`test_graph_training.py`](../prefill/tests/test_graph_training.py#L712-L741) and [`test_graph_training.py`](../prefill/tests/test_graph_training.py#L838-L949) |
| Evaluation reuses baseline helpers and namespaces graph tags | [`test_graph_eval.py`](../prefill/tests/test_graph_eval.py#L148-L172) |
| Checkpoint/dtype/model reconstruction is strict | [`test_graph_eval.py`](../prefill/tests/test_graph_eval.py#L175-L302) |
| Staged whole-context scoring preserves shapes and bounds hidden slices | [`test_graph_eval.py`](../prefill/tests/test_graph_eval.py#L305-L373) |
| Local-window rules and scoring cleanup match the contract | [`test_graph_eval.py`](../prefill/tests/test_graph_eval.py#L376-L446) |
| Evaluation restores the prefix and protects non-context tokens | [`test_graph_eval.py`](../prefill/tests/test_graph_eval.py#L449-L570) |

These are unit and integration tests with small models or fakes. They prove the
control flow and local mathematics, not Qwen3-8B throughput or long-context
resource use.

## 10. Known limitations and work still requiring a real pilot

1. **The planned one-context Qwen3-8B Slurm run has not been completed.** The
   required BGU cluster session was unavailable during implementation.
2. **Peak CPU RAM is not yet measured on a real long context.** CPU hidden
   states, gate-phase `u`, FAISS copies, and the teacher example can overlap.
3. **Peak GPU RAM and utilization are not yet validated for the target model.**
   Graph/joint processing and transient gate precomputation still materialize
   full graph-width `z/u` and edges on GPU.
4. **FAISS IVF compatibility, latency, and recall are not yet measured on the
   cluster build.** The default path repeatedly trains CPU indexes.
5. **The current code is single-device and ordinary-decoder only.** It does not
   support model sharding or hybrid/static hidden layouts.
6. **Evaluation tests use a fake runtime for the full orchestration loop.** A
   real checkpoint/LLM/dataset end-to-end run remains necessary.
7. **Reproducibility is context-boundary reproducibility, not environment
   capture.** Dataset revisions, tokenizer revisions, dependency versions,
   FAISS state, and Git commit are not stored in checkpoints.

## 11. Audited source map

The audit read the following production paths and their interactions:

- Entry points: [`train_graph.py`](../prefill/train_graph.py),
  [`eval_graph.py`](../prefill/eval_graph.py), and inherited
  [`eval.py`](../prefill/eval.py).
- Graph subsystem: [`builder.py`](../prefill/graph/builder.py),
  [`model.py`](../prefill/graph/model.py),
  [`training.py`](../prefill/graph/training.py), and
  [`evaluation.py`](../prefill/graph/evaluation.py).
- LLM/cache integration: [`attn.py`](../prefill/attention/attn.py),
  [`gate.py`](../prefill/attention/gate.py),
  [`kvcache.py`](../prefill/attention/kvcache.py),
  [`score.py`](../prefill/attention/score.py), and
  [`model/wrapper.py`](../prefill/model/wrapper.py).
- Prompt/data/evaluator helpers: [`data/wrapper.py`](../prefill/data/wrapper.py),
  [`template.py`](../prefill/model/template.py),
  [`tester.py`](../prefill/utils/tester.py), and
  [`func.py`](../prefill/utils/func.py).
- Operations: [`README.md`](../prefill/README.md) and
  [`requirements.txt`](../prefill/requirements.txt).
- Verification: [`test_graph_builder.py`](../prefill/tests/test_graph_builder.py),
  [`test_graph_model.py`](../prefill/tests/test_graph_model.py),
  [`test_graph_training.py`](../prefill/tests/test_graph_training.py),
  [`test_graph_train_cli.py`](../prefill/tests/test_graph_train_cli.py), and
  [`test_graph_eval.py`](../prefill/tests/test_graph_eval.py).

## Final mental model

The LLM is never trained. It produces causal context hidden states and
reconstruction targets one context at a time. The student turns each
layer/head's complete context into a graph, mixes the graph with GIN, maps that
mix back into hidden width, and lets the unchanged FastKVzip gate score it.
Two-phase mode gives the gate many token-minibatch updates and the graph side one
whole-context update. Evaluation recreates the same prefix and graph scorer,
scores only context positions, and feeds those scores into the repository's
existing retain-mask and generation benchmark.

The central boundary is therefore simple:

```text
LLM prefill and KVzip reconstruction are the frozen teacher.
GraphScorer is the trainable student.
RetainCache and Evaluator are the unchanged downstream consumer.
```
