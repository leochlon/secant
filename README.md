# RoPE attention is an exact forward-pass gradient step with softmax intact

Julie Huang<sup>1</sup>, Maggie Chlon<sup>1</sup>, Leon Chlon<sup>1,2</sup>

<sup>1</sup>Hassana Labs &nbsp;&nbsp; <sup>2</sup>Department of Information Engineering, Oxford University

[Paper (PDF)](rope_softmax_exact_construction.pdf) &nbsp;·&nbsp; [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/leochlon/secant/blob/main/RoPE_Softmax_Exact_Gradient_Step.ipynb)

## The identity

For every query $i$,

$$y_i = \mu_i + x_i^\top \Delta M_i, \qquad \Delta M_i = -\nabla_B \mathcal{E}_i(0),$$

with $\mu_i$ the uniform arithmetic mean of the permitted values and $\mathcal{E}_i(B)$ a weighted least-squares objective over the head's own rotated keys and centered values. Its regression weights, for attention scores $s_{ij}$ and softmax normalizer $Z_i$,

$$c_{ij} = \rho(s_{ij})/Z_i > 0, \qquad \rho(s) = (e^{s}-1)/s, \quad \rho(0) = 1,$$

are the divided difference of the exponential the softmax has already computed, so the attention write is exactly one gradient step from zero on that objective.

Softmax is not linearized and RoPE is not removed. The identity holds for arbitrary $W_q, W_k, W_v$ and affine projection biases, and extends to ALiBi, sign-preserving soft-caps, and Qwen3-style QK normalization.

## Contents

| File | |
| --- | --- |
| `rope_softmax_exact_construction.tex` | paper source |
| `rope_softmax_exact_construction.pdf` | compiled paper, 4 pp. |
| `RoPE_Softmax_Exact_Gradient_Step.ipynb` | verification notebook |

## Reproducing

Open the notebook in Colab, select a GPU runtime, Run all. It installs its own pinned dependencies (`transformers==4.51.3`, `numpy==2.1.3`, torch); there is no requirements file. The last cell exports a ZIP of measurements, exact replay factors, source, environment, and checksums.

| Profile | |
| --- | --- |
| `paper` (default) | Qwen2.5-0.5B, layer 23, 16 prefixes × 32 continuations, seed 0 |
| `quick` | same pipeline, 1 context × 4 queries |
| `smoke` | tiny random Qwen, no checkpoint download |

## Results

`paper` profile: Qwen2.5-0.5B, layer 23 (zero-indexed), 16 prefixes of 511 tokens, all 14 heads, 32 continuations at one position.

**Reconstruction.** All 240 context/head rows pass. The combined prefix and full-attention writes reconstruct at float64 MSE $9.78\times10^{-30}$ and $9.62\times10^{-30}$. Against the native FP32 write, MSE $7.01\times10^{-14}$ and maximum absolute error $1.4\times10^{-5}$.

**Frozen reuse.** Pooled MSE of the combined projected write when one query's matrix is reused across the other queries of its prefix.

| Predictor | Anchor 0 | All anchors |
| --- | ---: | ---: |
| Constant anchor output | 0.047787 | 0.045672 |
| Cache mean ($\Delta M = 0$) | 0.124573 | 0.124513 |
| Frozen anchor matrix | 0.324970 | 0.356763 |
| Linearized softmax | 0.161842 | 0.162068 |

Target RMS is 0.670545 for anchor 0 and 0.670824 across all anchors. Frozen rRMSE is 2.608 against anchor 0, 1.615 against the cache mean, and 2.795 all-anchor.

The matrix is strongly query-conditioned: freezing it is worse than reusing the anchor's output.

## Building the paper

```
pdflatex rope_softmax_exact_construction.tex   # twice, for cross-references
```

## Citation

```bibtex
@unpublished{huang2026rope,
  title  = {RoPE attention is an exact forward-pass gradient step with softmax intact},
  author = {Huang, Julie and Chlon, Maggie and Chlon, Leon},
  year   = {2026},
  note   = {\url{https://github.com/leochlon/secant}}
}
```
