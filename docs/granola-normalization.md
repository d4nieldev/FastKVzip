# GraNoLa in FastKVzip

This note describes the GraNoLa path introduced in PR #9. It focuses on two
questions:

1. How does a graph-aware GNN block update each token?
2. What exactly is \(\operatorname{Norm}(x_i)\)?

## 1. Inputs and implicit graph

Consider one layer–KV-head graph containing \(T\) tokens. Let

- \(x_i\in\mathbb{R}^{h}\) be token \(i\)'s raw mixer output;
- \(Y_1\in\mathbb{R}^{T\times d}\) be the mixer's first low-rank projection;
- \(d\) be `graph_dim`;
- \(c=T\) when token-count normalization is enabled, and \(c=1\) otherwise.

The graph's weighted adjacency is

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
u_i^{(0)}=[x_i\,\|\,\rho_i]\in\mathbb{R}^{h+r}.
\]

Later blocks receive the previous block's \(d\)-dimensional output.

## 2. One graph-aware GNN block

For GNN block \(\ell\), first project each token to width \(d\):

\[
p_i^{(\ell)}=W_{\ell,1}u_i^{(\ell-1)}.
\]

Then combine the token's own projection with the weighted projections of all
tokens:

\[
q_i^{(\ell)}
=p_i^{(\ell)}+\sum_{j=1}^{T}A_{ij}p_j^{(\ell)}.
\]

In matrix form,

\[
Q^{(\ell)}=P^{(\ell)}+AP^{(\ell)}.
\]

Equivalently, one complete block is

\[
\boxed{
U^{(\ell)}
=F_{\ell,2:m}\!\left((I+A)U^{(\ell-1)}W_{\ell,1}^{\top}\right)
},
\]

where \(F_{\ell,2:m}\) denotes the remaining LayerNorm–ReLU–Linear
layers. For \(m=1\), \(F_{\ell,2:m}\) is the identity function.

Because the first projection is linear, shared across nodes, and bias-free,

\[
(I+A)U^{(\ell-1)}W_{\ell,1}^{\top}
=\left((I+A)U^{(\ell-1)}\right)W_{\ell,1}^{\top}.
\]

Therefore this is exactly a generalized signed weighted GIN update on the
implicit complete graph—not merely “GIN-like.” There is one caveat: \(A\)
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

If `granola_mlp_depth` is \(m>1\), the remaining \(m-1\) layers apply

\[
v_t=W_{\ell,t}\,
\operatorname{ReLU}\!\left(\operatorname{LayerNorm}(v_{t-1})\right),
\qquad t=2,\ldots,m,
\]

starting with \(v_1=q_i^{(\ell)}\). The block output is
\(u_i^{(\ell)}=v_m\). When \(m=1\), the block output is simply
\(u_i^{(\ell)}=q_i^{(\ell)}\).

After \(k\) blocks, GraNoLa has a graph-aware token representation

\[
g_i=u_i^{(k)}\in\mathbb{R}^{d}.
\]

Two separate MLP heads map \(g_i\) to adaptive vectors
\(\gamma_i,\beta_i\in\mathbb{R}^{h}\):

\[
\gamma_i=f_\gamma(g_i),
\qquad
\beta_i=f_\beta(g_i).
\]

## 3. Exact formula for \(\operatorname{Norm}(x_i)\)

GraNoLa normalizes **each token across its \(h\) features**. For token \(i\),

\[
\mu_i=\frac{1}{h}\sum_{a=1}^{h}x_{i,a},
\]

\[
\sigma_i^2=\frac{1}{h}\sum_{a=1}^{h}(x_{i,a}-\mu_i)^2,
\]

and

\[
\boxed{
\operatorname{Norm}(x_i)_a
=\frac{x_{i,a}-\mu_i}{\sqrt{\sigma_i^2+10^{-5}}}
}
\qquad a=1,\ldots,h.
\]

This normalization has no trainable scale or bias. Instead, GraNoLa predicts
them from the graph representation and applies

\[
\widehat{x}_i
=\gamma_i\odot\operatorname{Norm}(x_i)+\beta_i.
\]

The mixer contribution is then

\[
\Delta_i
=\alpha\,\operatorname{LeakyReLU}(\widehat{x}_i).
\]

The key distinction from the BatchNorm path is that GraNoLa's mean and
variance are computed across the features of one token, while its affine
vectors \(\gamma_i\) and \(\beta_i\) are generated separately for every token
using graph context.

> **Parameter-counting detail:** the LayerNorms inside the GNN and output
> heads have learned scale and bias. The \(\operatorname{Norm}(x_i)\) operation
> above does not; its affine transformation is supplied by the predicted
> \(\gamma_i\) and \(\beta_i\).

## 4. Is this mathematically identical to the paper?

**No—not as a complete architecture.** It preserves the paper's outer
GraNoLa structure, but replaces the graph and the normalization GNN.

The paper defines

\[
Z=\operatorname{GNN}_{\mathrm{NORM}}(A,[X\,\|\,R]),
\qquad
\gamma=f_1(Z),\quad \beta=f_2(Z),
\]

followed by

\[
\gamma_i\odot\operatorname{Norm}(x_i)+\beta_i.
\]

FastKVzip follows this same template. The mathematical changes are inside
\(A\) and \(\operatorname{GNN}_{\mathrm{NORM}}\):

| Part | Paper | FastKVzip PR #9 |
|---|---|---|
| Graph | Given input adjacency \(A\) | Learned complete low-rank adjacency \(A=Y_1Y_1^\top/c\) |
| Aggregation | General RNF-augmented MPNN; experiments use standard GIN/GINE | Generalized weighted GIN: \(P+AP\) |
| Width | Experimental GNN and RNF widths match feature width | Configurable `graph_dim` and RNF width, usually much smaller than \(h\) |
| RNF | Fresh random features in the experimental formulation | Fresh during training, deterministic per example during evaluation |
| Sharing | A normalization module for each backbone GNN layer | Per layer–head graph, per layer, or globally shared |
| Output wrapper | Normalized features followed by the backbone's activation | Learned \(\alpha\), LeakyReLU, and residual addition to the gate input |

The per-token normalization and adaptive \(\gamma_i,\beta_i\) are faithful to
the paper. Each FastKVzip block is also exactly a weighted GIN operation on its
own implicit graph. What differs from the paper's experimental GIN/GINE is the
graph and message definition: FastKVzip uses dense, signed, data-dependent
scalar weights and no edge-feature messages. It is therefore a **GraNoLa
adaptation using generalized weighted GIN**, rather than the same experimental
GIN/GINE instantiation.

Because the graph operator and RNF setup changed, the paper's full-adaptivity
and universality results do not automatically apply to this implementation.

## Implementation references

- [Implicit GraNoLa GNN update](https://github.com/d4nieldev/FastKVzip/blob/3e365f82b5a4f2d62ec0ca050ecb4ebe89436765/prefill/graph/model.py#L595-L666)
- [Per-token normalization](https://github.com/d4nieldev/FastKVzip/blob/3e365f82b5a4f2d62ec0ca050ecb4ebe89436765/prefill/graph/model.py#L561-L567)
- [Adaptive gamma and beta application](https://github.com/d4nieldev/FastKVzip/blob/3e365f82b5a4f2d62ec0ca050ecb4ebe89436765/prefill/graph/model.py#L762-L808)
- [Paper, Section 3.2 and Equations 8–10](https://arxiv.org/html/2404.13344v2#S3.SS2)
- [PR statement describing this as an adaptation](https://github.com/d4nieldev/FastKVzip/blob/3e365f82b5a4f2d62ec0ca050ecb4ebe89436765/prefill/README.md#L80-L83)
