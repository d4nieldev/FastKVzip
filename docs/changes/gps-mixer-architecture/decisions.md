# Implementation decisions — GPS mixer architecture

Choices made while implementing [the approved plan](plan.md) that the plan left
open or that departed from it.

| Marker and label | Meaning |
| --- | --- |
| 🔴 Plan deviation | Contradicts the approved plan without separate approval. |
| 🟠 Plan gap | The plan left this code choice open. |
| 🟢 User-approved amendment | Approved separately, after the plan. |
| ⚪ Superseded | Replaced by a later decision. |

## D1 — 🟠 Plan gap — the GPS block normalizes per token, not over the context

- **Background:**
  - The plan says each GPS branch gets "its own residual connection and normalization" but does not say which kind.
  - The existing mixer normalizes over the whole context, which needs a separate pass to collect statistics before any token can be scored.
- **Decision:** GPS normalizes each token across its own features, with a learned scale and shift per graph.
- **Plan gap or deviation:** The plan named the normalization's position in the block but not its type.
- **Reason and tradeoff:**
  - Per-token normalization needs no statistics pass, so a block can be stacked without adding a pass per block.
  - The cost is that GPS and the implicit mixer no longer normalize the same way, so a measured difference between them is not purely the branch structure.

| Code reference | What this code does |
| --- | --- |
| [model.py:435-457](../../../prefill/graph/model.py#L435-L457) | Normalizes each token over its features, scaled and shifted per graph. |
| [model.py:622-666](../../../prefill/graph/model.py#L622-L666) | Places one such normalization after each branch and after the feedforward. |

## D2 — 🟠 Plan gap — the feedforward width is fixed rather than exposed

- **Background:**
  - The plan puts a feedforward block at graph width and lists the settings to expose: architecture, depth, attention heads, and random-feature count.
  - A feedforward block also has an inner width, which the plan does not mention.
- **Decision:** The inner width is twice the graph width and is not a setting.
- **Plan gap or deviation:** The plan neither exposed this width nor fixed it.
- **Reason and tradeoff:**
  - Graph width already controls the block's size, so a second width knob adds experiment surface for little gain.
  - Changing it later means a new setting and a new checkpoint field, which is why the value is named in one place.

| Code reference | What this code does |
| --- | --- |
| [model.py:33-36](../../../prefill/graph/model.py#L33-L36) | Fixes the multiplier in one place, with the reason. |

## D3 — 🟠 Plan gap — GPS drops the hidden-width scale and shift

- **Background:**
  - The implicit mixer normalizes, then applies a learned scale and shift at hidden width, then the activation.
  - The plan says GPS ends with "the same activation as today," without saying whether that scale and shift come with it.
- **Decision:** GPS projects back to hidden width, applies the activation, and scales by the residual weight, with no hidden-width scale and shift.
- **Plan gap or deviation:** The plan's step 7 was ambiguous about which parts of the final stage GPS keeps.
- **Reason and tradeoff:**
  - The last block already applies a learned scale and shift in the graph space, so repeating it at hidden width is redundant.
  - It also saves two parameters per graph per hidden unit, which is the expensive direction here.

| Code reference | What this code does |
| --- | --- |
| [model.py:762-773](../../../prefill/graph/model.py#L762-L773) | Projects to hidden width, applies the activation, scales by the residual weight. |

## D4 — 🟠 Plan gap — the random-feature draw needed a uniformity correction

- **Background:**
  - Performer's accuracy depends on drawing feature directions uniformly over all directions.
  - Building them from a standard matrix factorization looks correct but is not uniform, because the factorization's sign convention favors some directions.
- **Decision:** The draw folds in the factorization's signs, restoring uniformity.
- **Plan gap or deviation:** The plan specified Performer but not how its features are drawn.
- **Reason and tradeoff:**
  - Without the correction the attention did not converge to softmax attention: its error stalled near 10% however many features were added.
  - With it the error falls as features grow, which is the property this branch is supposed to have.

| Code reference | What this code does |
| --- | --- |
| [model.py:481-500](../../../prefill/graph/model.py#L481-L500) | Draws orthogonal directions and corrects their signs. |
| [test_gps_mixer.py:83-104](../../../prefill/tests/test_gps_mixer.py#L83-L104) | Fails if the estimate stops converging to the exact kernel. |

## D5 — 🟠 Plan gap — queries and keys are stabilized differently

- **Background:**
  - Performer exponentiates a score before summing, so a large score overflows and a small one underflows.
  - Subtracting a maximum first is safe whenever that maximum is constant across the summed axis, because it then cancels between the attention's numerator and denominator.
- **Decision:** Queries subtract a maximum taken per token; keys subtract one taken over the whole subgraph.
- **Plan gap or deviation:** The plan did not cover numerical conditioning of the branch.
- **Reason and tradeoff:**
  - A single subgraph-wide maximum for queries makes a token whose scores sit far below it underflow, which training does in bfloat16.
  - Per-token maxima for queries avoid that, and the branch now tracks its double-precision result to within 1.5% in bfloat16.

| Code reference | What this code does |
| --- | --- |
| [model.py:551-565](../../../prefill/graph/model.py#L551-L565) | Chooses the axis the maximum is taken over. |
| [test_gps_mixer.py:134-145](../../../prefill/tests/test_gps_mixer.py#L134-L145) | Fails if the bfloat16 result stops being finite or drifts. |

## D6 — 🟠 Plan gap — GPS rejoins streamed token chunks instead of refusing them

- **Background:**
  - Training and validation hand the mixer a span in token-sized pieces, because the implicit mixer can summarize a context piece by piece.
  - A GPS stack cannot consume a span piecewise: its attention and normalization both span the whole subgraph.
- **Decision:** GPS accepts the pieces and rejoins them before running.
- **Plan gap or deviation:** The plan restricted GPS to subgraphs but did not say what happens at this internal streaming entry point, which validation uses every epoch.
- **Reason and tradeoff:**
  - Refusing it would have made GPS training crash at its first validation, since callers reach the mixer this way whether or not they subdivide.
  - The span is a bounded subgraph, so rejoining costs nothing that scoring it did not already cost.

| Code reference | What this code does |
| --- | --- |
| [model.py:794-820](../../../prefill/graph/model.py#L794-L820) | Rejoins the pieces and runs the stack once. |
| [test_gps_mixer.py:251-272](../../../prefill/tests/test_gps_mixer.py#L251-L272) | Checks piecewise delivery gives the same result, and that a short span is rejected. |
| [test_gps_mixer.py:231-248](../../../prefill/tests/test_gps_mixer.py#L231-L248) | Scores a held-out context the way validation does. |

## D7 — 🟠 Plan gap — weight decay is chosen by the architecture

- **Background:**
  - The optimizer applied weight decay to the implicit mixer's two projections and exempted its scale, shift, and residual weight, all named directly.
  - GPS has many more parameters, including three normalizations per block.
- **Decision:** Each architecture reports its own decayed and undecayed groups, and the optimizer uses whatever it reports.
- **Plan gap or deviation:** The plan did not say how the optimizer treats the new parameters.
- **Reason and tradeoff:**
  - Naming GPS parameters in the optimizer would need updating whenever the block changes, and a missed name silently decays a normalization.
  - Both architectures keep the same convention: normalization scales and shifts and the residual weight stay undecayed.

| Code reference | What this code does |
| --- | --- |
| [training.py:325-329](../../../prefill/graph/training.py#L325-L329) | Asks the mixer for its groups instead of naming parameters. |
| [model.py:747-760](../../../prefill/graph/model.py#L747-L760) | Classifies every GPS parameter, so a new one cannot be missed. |

## D8 — 🟠 Plan gap — no automatic parameter matching between architectures

- **Background:**
  - The plan notes the two architectures will not have equal parameter counts at the same graph width.
  - It says counts "get reported at startup" but not whether anything adjusts them.
- **Decision:** Nothing is adjusted automatically; both training scripts print the mixer's parameter count.
- **Plan gap or deviation:** The plan left open whether matching should be automatic.
- **Reason and tradeoff:**
  - Matching counts automatically would silently override the graph width someone asked for.
  - The cost is that a like-for-like comparison needs the widths chosen by hand.

| Code reference | What this code does |
| --- | --- |
| [train_graph.py:1114-1121](../../../prefill/train_graph.py#L1114-L1121) | Prints the count for teacher-supervised runs. |
| [train_graph_answer.py:1442-1449](../../../prefill/train_graph_answer.py#L1442-L1449) | Prints the same for answer-supervised runs. |

## D9 — 🟠 Plan gap — gate-only runs keep no architecture

- **Background:**
  - A run can train the gate alone, with no mixer at all, which was added shortly before this work.
  - Such a run has no mixer for an architecture to describe.
- **Decision:** A gate-only run records the default architecture and rejects the GPS settings, as it already rejects the other mixer-only settings.
- **Plan gap or deviation:** The plan predates the gate-only mode and does not mention it.
- **Reason and tradeoff:**
  - Rejecting them keeps the existing promise that a setting someone chose is never silently ignored.
  - The alternative, accepting and dropping them, would put a GPS setting in the checkpoint that nothing applied.

| Code reference | What this code does |
| --- | --- |
| [model.py:1022-1025](../../../prefill/graph/model.py#L1022-L1025) | Builds no mixer when there is no graph width, whatever the architecture. |
| [train_graph_answer.py:76-88](../../../prefill/train_graph_answer.py#L76-L88) | Adds the GPS settings to those a gate-only run rejects. |
