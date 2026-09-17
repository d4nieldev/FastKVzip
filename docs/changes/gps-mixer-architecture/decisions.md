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
| [model.py:630-674](../../../prefill/graph/model.py#L630-L674) | Places one such normalization after each branch and after the feedforward. |

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
| [model.py:770-788](../../../prefill/graph/model.py#L770-L788) | Projects to hidden width, applies the activation, scales by the residual weight. |
| [test_gps_mixer.py:555-573](../../../prefill/tests/test_gps_mixer.py#L555-L573) | Fails if the activation or its slope stops being applied. |

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
| [test_gps_mixer.py:92-113](../../../prefill/tests/test_gps_mixer.py#L92-L113) | Fails if the estimate stops converging to the exact kernel. |

## D5 — 🟠 Plan gap — the attention branch's numerical conditioning

- **Background:**
  - Performer exponentiates a score before summing, so a large score overflows and a small one underflows.
  - Subtracting a maximum first is safe whenever that maximum is constant across the summed axis, because it then cancels between the attention's numerator and denominator.
- **Decision:**
  - Queries subtract a maximum taken per token; keys subtract one taken over the whole subgraph.
  - The denominator is floored only at the smallest representable value, not at a fixed constant.
- **Plan gap or deviation:** The plan did not cover numerical conditioning of the branch.
- **Reason and tradeoff:**
  - A single subgraph-wide maximum for queries makes a token whose scores sit far below it underflow, which training does in bfloat16.
  - Those same stabilizers leave the denominator on no fixed scale, so an absolute floor clamped healthy values and quietly shrank the output; every feature is a positive exponential, so only underflow needs guarding.

| Code reference | What this code does |
| --- | --- |
| [model.py:551-565](../../../prefill/graph/model.py#L551-L565) | Chooses the axis each maximum is taken over. |
| [model.py:567-590](../../../prefill/graph/model.py#L567-L590) | Floors the denominator at the dtype's smallest value. |
| [test_gps_mixer.py:498-515](../../../prefill/tests/test_gps_mixer.py#L498-L515) | Fails if the stabilizer stops preventing underflow. |

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
| [model.py:809-835](../../../prefill/graph/model.py#L809-L835) | Rejoins the pieces and runs the stack once. |
| [test_gps_mixer.py:280-301](../../../prefill/tests/test_gps_mixer.py#L280-L301) | Checks piecewise delivery gives the same result, and that a short span is rejected. |
| [test_gps_mixer.py:246-277](../../../prefill/tests/test_gps_mixer.py#L246-L277) | Scores a held-out context the way validation does. |

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
| [model.py:755-768](../../../prefill/graph/model.py#L755-L768) | Classifies every GPS parameter, so a new one cannot be missed. |
| [test_gps_mixer.py:585-596](../../../prefill/tests/test_gps_mixer.py#L585-L596) | Fails if a normalization starts being decayed. |

## D8 — 🟠 Plan gap — the mixer's correction is returned at a fixed precision

- **Background:**
  - Runs use bfloat16, but the gate's own normalization always returns a higher precision.
  - The implicit mixer returns its correction in that higher precision too, and adding it is what lifts the gate's input to match.
- **Decision:** GPS returns its correction at the same precision the implicit mixer does, and its local branch accumulates its sum over tokens there as well.
- **Plan gap or deviation:**
  - The plan said the gate "receives the same kind of correction it receives now" without stating that precision is part of that contract.
  - Built to the plan as written, GPS returned bfloat16 and every run failed on its first scored subgraph.
- **Reason and tradeoff:**
  - Matching the existing contract is what makes GPS usable at all in the precision runs actually use.
  - It costs a little memory over keeping the whole stack in bfloat16, which is the same trade the implicit mixer already makes.

| Code reference | What this code does |
| --- | --- |
| [model.py:770-788](../../../prefill/graph/model.py#L770-L788) | Returns the correction at the reduction precision. |
| [model.py:619-627](../../../prefill/graph/model.py#L619-L627) | Accumulates the local branch's sum over tokens the same way. |
| [test_gps_mixer.py:599-605](../../../prefill/tests/test_gps_mixer.py#L599-L605) | Fails if the correction is left in the compute precision. |

## D9 — 🟠 Plan gap — a checkpoint that predates the choice is told which architecture it holds

- **Background:**
  - Resuming compares the saved settings against the settings the new run would write, and any difference stops the run.
  - Checkpoints written before this change record no architecture at all.
- **Decision:** Both training scripts fill in the implicit architecture and the GPS defaults for such a checkpoint before anything reads it.
- **Plan gap or deviation:**
  - The plan promised resume was untouched and that old checkpoints load as implicit, but not how.
  - Adding the four settings without this made every pre-change checkpoint fail to resume.
- **Reason and tradeoff:**
  - Naming the architecture also makes warm-starting an old checkpoint under GPS fail immediately with a clear message, rather than later when implicit weights will not load into a GPS stack.
  - Dropping the new keys from the comparison instead would have resumed cleanly but left that mismatch to surface as a confusing load error.

| Code reference | What this code does |
| --- | --- |
| [train_graph.py:291-298](../../../prefill/train_graph.py#L291-L298) | Fills the settings in for a teacher-supervised resume. |
| [train_graph_answer.py:349-364](../../../prefill/train_graph_answer.py#L349-L364) | Does the same for the answer-supervised scripts, skipping gate-only checkpoints. |
| [test_gps_mixer.py:891-912](../../../prefill/tests/test_gps_mixer.py#L891-L912) | Fails if a pre-change checkpoint would conflict on resume. |
| [test_gps_mixer.py:951-972](../../../prefill/tests/test_gps_mixer.py#L951-L972) | Fails if a warm start could switch architecture. |

## D10 — 🟠 Plan gap — settings that belong to one architecture are refused under the other

- **Background:**
  - A run can train the gate alone, with no mixer, and it already refuses every mixer-only setting rather than ignoring it.
  - Depth, attention heads, and random-feature count apply only to GPS.
- **Decision:** Passing any of them without choosing GPS is an error, and a gate-only run refuses them along with the other mixer settings.
- **Plan gap or deviation:** The plan said depth applies to GPS only but did not say what happens when it is set anyway.
- **Reason and tradeoff:**
  - Accepting them would have written a setting into the checkpoint that nothing applied, which is exactly what the existing gate-only rule forbids.
  - The cost is that switching a command line to the implicit architecture now also means removing those settings.

| Code reference | What this code does |
| --- | --- |
| [train_graph.py:383-388](../../../prefill/train_graph.py#L383-L388) | Refuses them under another architecture. |
| [train_graph_answer.py:608-613](../../../prefill/train_graph_answer.py#L608-L613) | Does the same for answer-supervised runs. |
| [train_graph_answer.py:79-88](../../../prefill/train_graph_answer.py#L79-L88) | Adds them to the settings a gate-only run refuses. |
| [test_gps_mixer.py:975-986](../../../prefill/tests/test_gps_mixer.py#L975-L986) | Fails if they are silently accepted. |

## D11 — 🟠 Plan gap — the recorded activation order names the architecture

- **Background:**
  - A checkpoint records the order its activation is applied in, and loading validates it.
  - GPS applies a different sequence from the implicit mixer.
- **Decision:** Each architecture records its own activation order, and loading checks the one belonging to the architecture the checkpoint claims.
- **Plan gap or deviation:** The plan listed four settings to record and did not mention this existing field, which the new architecture changes the meaning of.
- **Reason and tradeoff:**
  - It gives a second, independent reason a GPS checkpoint cannot be read as implicit, beyond the settings themselves.
  - It is one more field to keep in step if either block's activation sequence ever changes.

| Code reference | What this code does |
| --- | --- |
| [model.py:27](../../../prefill/graph/model.py#L27) | Names the GPS order. |
| [test_gps_mixer.py:423-455](../../../prefill/tests/test_gps_mixer.py#L423-L455) | Round-trips the config a real run writes, so neither field can go missing. |

## D12 — 🔴 Plan deviation — one promised check was replaced, because it cannot hold

- **Background:**
  - The plan's verification list includes "GPS reduces to the current mechanism when its global branch is zeroed."
  - That would hold only if GPS were the implicit mixer plus an attention term.
- **Decision:** That check is not implemented. Its purpose — confirming the branches are wired as described — is covered by checks that each piece of the block is load-bearing and that each graph uses its own weights.
- **Plan gap or deviation:** A verification the plan promised is absent, so it is recorded rather than quietly dropped.
- **Reason and tradeoff:**
  - GPS also normalizes per token (D1), adds a feedforward, and adds a position encoding, so zeroing the attention branch does not recover the implicit mixer; the check would fail on correct code.
  - The replacement checks are narrower per test but, unlike the promised one, they actually fail when a piece of the block is removed.

| Code reference | What this code does |
| --- | --- |
| [test_gps_mixer.py:375-410](../../../prefill/tests/test_gps_mixer.py#L375-L410) | Fails if any per-graph weight stops being selected per graph. |
| [test_gps_mixer.py:472-495](../../../prefill/tests/test_gps_mixer.py#L472-L495) | Fails if evaluation rebuilds a GPS checkpoint as something else. |

## D13 — 🟠 Plan gap — no automatic parameter matching between architectures

- **Background:**
  - The plan notes the two architectures will not have equal parameter counts at the same graph width.
  - It says counts are reported at startup but not whether anything adjusts them.
- **Decision:** Nothing is adjusted automatically; both training scripts print the mixer's parameter count.
- **Plan gap or deviation:** The plan left open whether matching should be automatic.
- **Reason and tradeoff:**
  - Matching counts automatically would silently override the graph width someone asked for.
  - The cost is that a like-for-like comparison needs the widths chosen by hand.

| Code reference | What this code does |
| --- | --- |
| [train_graph.py:1138-1145](../../../prefill/train_graph.py#L1138-L1145) | Prints the count for teacher-supervised runs. |
| [train_graph_answer.py:1494-1501](../../../prefill/train_graph_answer.py#L1494-L1501) | Prints the same for answer-supervised runs. |
