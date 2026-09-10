# Offline verification of the identity

The repository ships no measurement artifacts — every notebook cell has
`execution_count: null` and there is no CSV, JSON or npz anywhere — so §6's
numbers cannot be checked without running the notebook end to end against a
downloaded checkpoint. These three files let the identity be checked in seconds,
with NumPy alone, no model and no network.

```
python replication/verify_identity.py replication/reference_capture.npz
```

## What is captured

`reference_capture.npz` (1.0 MB) follows the `paper` profile: post-RoPE queries,
keys and values at **layer 23** of **Qwen2.5-0.5B** at the revision the notebook
pins, over **511-token** prefixes, all **14 query heads** and both KV groups, in
the model's own float32. Two contexts, which is enough to exercise the identity
and the gauge family; `capture_reference.py` regenerates it and takes
`--contexts` if you want more.

Storing float32 and verifying in float64 is deliberate: it is the same
arrangement the paper's float64 residuals come from, since the native tensors
are float32 either way.

## What it reports

Measured on the capture in this directory:

| Check | Result |
| --- | ---: |
| `max │y_direct − (μ + qᵀΔM)│` | **7.88e−14** |
| `max │⟨∇E(0) + ΔM, u⟩│` over random directions | **6.75e−12** |
| gauge freedom, κ ∈ {0, 1, 2, −3} | all reproduce y to ~1e−14 |
| unit step increases E | **3 of 28 heads (11%)** |
| λ_max(H) | median 0.51, max 109.56 |

The first two are the paper's claims and they hold. The gauge row is the
paper's own eq. `gauge` made concrete: κ = 1 is a choice, not a necessity, and
the identity is indifferent to it.

The last two rows are a separate question from whether ΔM equals −∇E(0), which
it does. A unit-step gradient step reduces a quadratic only where λ_max(H) < 2,
so "one gradient step" is a claim about the gradient, not about descent. At
layer 23 the step does descend in the large majority of heads. It is worth
knowing that this is a property of the layer rather than of the construction:
pooled across all 24 layers of a same-size Qwen2.5 checkpoint the step increases
E in 56% of heads, driven almost entirely by early layers where λ_max reaches
6–21, while layers 16 and above sit near zero.

## Provenance

An independent re-capture, not the authors' original file. Produced on macOS
arm64, Python 3.12, torch 2.14.0, transformers 4.57.3, numpy 2.5.3, CPU, from
`Qwen/Qwen2.5-0.5B` revision `060db6499f32faf8b98477b0a26969ef7d8b9987`. Note
that the notebook pins transformers 4.51.3; the capture is of projection
outputs and rotary application, which did not change between those versions,
but regenerating under the pinned version is a one-line change if you would
rather.

Arrays load with `allow_pickle=False`. The npz doubles the size of the
repository — if that is unwelcome, the two scripts alone are useful and
`capture_reference.py` reproduces the file in about a minute.
