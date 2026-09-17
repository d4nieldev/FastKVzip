# GraNoLa in FastKVzip

This note describes the GraNoLa path in PR #9. It answers three questions:

1. What does the normalization GNN read, and what does it produce?
2. What exactly is \(\operatorname{Norm}(\cdot)\), and at which width?
3. How far is this from the GRANOLA paper?

## 1. Inputs and implicit graph

Consider one layer–KV-head graph over \(T\) context tokens. The mixer projects
the context hidden states \(X\in\mathbb{R}^{T\times h}\) once:

\[
Y_1=XW_a,\qquad R=Y_2=XW_v,\qquad Y_1,R\in\mathbb{R}^{T\times d},
\]

where \(d\) is `graph_dim`. Write \(c=T\) when token-count normalization is
enabled and \(c=1\) otherwise. The graph's weighted adjacency is

\[
A=\frac{Y_1Y_1^\top}{c},
\qquad
A_{ij}=\frac{\langle Y_{1,i},Y_{1,j}\rangle}{c}.
\]

The weights can be positive or negative. The implementation never constructs
the full \(T\times T\) matrix \(A\).

GraNoLa also samples a non-trainable random node feature
\(\rho_i\in\mathbb{R}^{r}\) for each token. The first GNN block receives

\[
u_i^{(0)}=[R_i\,\|\,\rho_i]\in\mathbb{R}^{d+r}.
\]

Later blocks receive the previous block's \(d\)-dimensional output. Everything
the normalization GNN touches is therefore graph width, never hidden width.

Each subgraph is its own graph, so the RNF draw is keyed on the graph identity
**and** the subgraph's offset in the context. Two blocks of one layer–head pair
would otherwise share a draw and become indistinguishable to the GNN.

## 2. One graph-aware GNN block

For GNN block \(\ell\), first project each token to width \(d\):

\[
p_i^{(\ell)}=W_{\ell,1}u_i^{(\ell-1)}.
\]

Then combine the token's own projection with the weighted projections of all
tokens:

\[
q_i^{(\ell)}
=p_i^{(\ell)}+\sum_{j=1}^{T}A_{ij}p_j^{(\ell)},
\qquad
Q^{(\ell)}=P^{(\ell)}+AP^{(\ell)}.
\]

Equivalently, one complete block is

\[
\boxed{
U^{(\ell)}
=F_{\ell,2:m}\!\left((I+A)U^{(\ell-1)}W_{\ell,1}^{\top}\right)
},
\]

where \(F_{\ell,2:m}\) denotes the remaining LayerNorm–ReLU–Linear layers. For
\(m=1\), \(F_{\ell,2:m}\) is the identity function.

Because the first projection is linear, shared across nodes, and bias-free,

\[
(I+A)U^{(\ell-1)}W_{\ell,1}^{\top}
=\left((I+A)U^{(\ell-1)}\right)W_{\ell,1}^{\top}.
\]

Therefore this is exactly a generalized signed weighted GIN update on the
implicit complete graph—not merely "GIN-like." There is one caveat: \(A\)
contains its diagonal, so

\[
q_i=(1+A_{ii})p_i+\sum_{j\ne i}A_{ij}p_j.
\]

The explicit self coefficient is \(1\), but the total self coefficient is
\(1+A_{ii}\). Standard GIN normally uses an unweighted neighbor sum and a
single fixed or learned scalar \(1+\epsilon\).

The update is evaluated without forming \(A\):

\[
C^{(\ell)}=\frac{Y_1^\top P^{(\ell)}}{c},
\qquad
AP^{(\ell)}=Y_1C^{(\ell)}.
\]

After \(k\) blocks, GraNoLa has a graph-aware token representation
\(g_i=u_i^{(k)}\in\mathbb{R}^{d}\).

## 3. Readout, normalization and the affine parameters

`--granola-adaptivity` selects between two coupled presets. The choice fixes
both the readout and the normalization statistic, because the two only make
sense together.

### `graph` (the default)

The readout is a mean over tokens, so each graph gets one affine pair:

\[
z=\frac{1}{T}\sum_{i=1}^{T}g_i,
\qquad
\gamma=f_\gamma(z),\quad \beta=f_\beta(z),
\qquad
\gamma,\beta\in\mathbb{R}^{d}.
\]

The normalized quantity is the aggregated message \(AR\), standardized over the
**token** axis per feature — the same statistic the BatchNorm branch uses, at
graph width:

\[
\mu_a=\frac{1}{T}\sum_{i=1}^{T}(AR)_{i,a},
\qquad
\sigma_a^2=\frac{1}{T}\sum_{i=1}^{T}\left((AR)_{i,a}-\mu_a\right)^2 .
\]

Both \(z\) and \((\mu,\sigma)\) summarize the whole context. They are computed
once when the graph is prepared and are never recomputed from a token chunk, so
the token and graph microbatch splits cannot move the scores.

### `token`

No readout: the affine parameters are per token, as in the paper.

\[
\gamma_i=f_\gamma(g_i),\qquad \beta_i=f_\beta(g_i),
\qquad
\gamma_i,\beta_i\in\mathbb{R}^{d}.
\]

The statistic is then LayerNorm-node over the \(d\) features of each token,
matching the paper's Equation 6:

\[
\mu_i=\frac{1}{d}\sum_{a=1}^{d}(AR)_{i,a},
\qquad
\sigma_i^2=\frac{1}{d}\sum_{a=1}^{d}\left((AR)_{i,a}-\mu_i\right)^2 .
\]

### Why the two are coupled

LayerNorm-node strips each token's magnitude: every normalized token has the
same length. In the paper that is harmless, because per-node \(\gamma_i,\beta_i\)
put node-specific scale back, and that is the mechanism. A single per-graph
affine pair cannot, so pairing `graph` adaptivity with LayerNorm-node would
discard the per-token magnitude that the retention gate ranks on, with nothing
to restore it. Standardizing over tokens instead removes only a per-feature
mean and scale, and the differences between tokens survive.

### The update

\[
\widehat{x}=\gamma\odot\operatorname{Norm}(AR)+\beta,
\qquad
\Delta=\alpha\,\operatorname{LeakyReLU}\!\left(\widehat{x}\,W_o\right).
\]

The affine is graph width, so the out projection \(W_o\) that the BatchNorm
branch folds into its kernel is applied here instead. The leaky ReLU and the
residual still happen at hidden width, exactly as in the BatchNorm branch.

> **Parameter-counting detail:** the LayerNorms inside the GNN and the output
> heads have learned scale and bias. The \(\operatorname{Norm}\) operation above
> does not; its affine transformation is the predicted \(\gamma\) and \(\beta\).

## 4. Is this mathematically identical to the paper?

**No—not as a complete architecture.** It preserves the paper's outer GraNoLa
structure, but replaces the graph, the normalization GNN, and (by default) the
granularity of the affine parameters.

The paper defines

\[
Z=\operatorname{GNN}_{\mathrm{NORM}}(A,[X\,\|\,R]),
\qquad
\gamma=f_1(Z),\quad \beta=f_2(Z),
\]

followed by \(\gamma_i\odot\operatorname{Norm}(x_i)+\beta_i\).

FastKVzip follows this template. The mathematical changes are:

| Part | Paper | FastKVzip PR #9 |
|---|---|---|
| Graph | Given input adjacency \(A\) | Learned complete low-rank adjacency \(A=Y_1Y_1^\top/c\) |
| GNN input | Backbone features \(X\) concatenated with RNF | Message features \(R=XW_v\) concatenated with RNF |
| Aggregation | General RNF-augmented MPNN; experiments use standard GIN/GINE | Generalized weighted GIN: \(P+AP\) |
| Width | Experimental GNN and RNF widths match feature width | `graph_dim` and RNF width, much smaller than \(h\) |
| Affine granularity | Per node | Per graph by default; per token under `--granola-adaptivity token` |
| Statistic | LayerNorm-node | Over tokens per feature by default; LayerNorm-node under `token` |
| RNF | Fresh random features | Fresh during training, deterministic per example during evaluation, keyed per subgraph |
| Sharing | A normalization module for each backbone GNN layer | Per layer–head graph, per layer, or globally shared |
| Output wrapper | Normalized features followed by the backbone's activation | Out projection, learned \(\alpha\), LeakyReLU, residual addition to the gate input |

`--granola-adaptivity token` is the setting closest to the paper: per-node
affine parameters over a LayerNorm-node statistic. It still differs in the
graph, the GNN input and the width.

Because the graph operator, the RNF setup and the default affine granularity
changed, the paper's full-adaptivity and universality results do not
automatically apply to this implementation.

## Implementation references

- `granola_gnn` and `_prepare_granola` in `prefill/graph/model.py`
- `normalized`, `granola_affine`, `activated` and `projected_activation` in the
  same file
- `_train_granola_batch` in `prefill/graph/training.py`
- [Paper, Section 3.2 and Equations 8–10](https://arxiv.org/html/2404.13344v2#S3.SS2)
