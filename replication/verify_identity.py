#!/usr/bin/env python3
"""Check the identity on a captured layer. NumPy only; no model, no network.

    y_i = mu_i + x_i^T dM_i,   dM_i = -grad_B E_i(0),   c_ij = rho(s_ij)/Z_i

Reports three things:
  1. the residual against attention computed directly, per head;
  2. the gauge freedom -- any constant kappa reproduces y, not only kappa = 1;
  3. whether the unit step actually descends E, which is a separate question
     from whether dM equals -grad E(0).
"""
import argparse, json
import numpy as np


def rho(s):
    """Divided difference of exp between 0 and s: (e^s - 1)/s, with rho(0) = 1."""
    out = np.empty_like(s)
    small = np.abs(s) < 1e-8
    out[small] = 1.0 + s[small] / 2.0
    tail = s[~small]
    out[~small] = np.expm1(tail) / tail
    return out


def weights(s, kappa=1.0):
    """c_ij = (e^{s_ij} - kappa)/(s_ij Z_i), computed without overflowing."""
    peak = s.max(axis=-1, keepdims=True)
    shifted = np.exp(s - peak)
    normalizer = shifted.sum(axis=-1, keepdims=True)          # Z = e^peak * normalizer
    numerator = shifted - kappa * np.exp(-peak)
    safe = np.where(np.abs(s) < 1e-12, 1.0, s)
    return np.where(np.abs(s) < 1e-12, np.exp(-peak) / normalizer,
                    numerator / (safe * normalizer)), shifted / normalizer


def check(path):
    data = np.load(path, allow_pickle=False)
    meta = json.loads(bytes(data["meta"]).decode())
    groups = meta["heads"] // meta["kv_heads"]
    print(json.dumps(meta, indent=1))
    worst_identity = worst_gradient = 0.0
    increases = total = 0
    lambdas = []

    for context in range(meta["contexts"]):
        query = data[f"context_{context}_query"].astype(np.float64)         # H, D
        keys = data[f"context_{context}_keys"].astype(np.float64)           # G, N, D
        values = data[f"context_{context}_values"].astype(np.float64)
        keys, values = np.repeat(keys, groups, 0), np.repeat(values, groups, 0)
        scores = np.einsum("hd,hnd->hn", query, keys) * meta["scale"]
        coefficients, probabilities = weights(scores)
        direct = np.einsum("hn,hnd->hd", probabilities, values)             # softmax attention
        mu = values.mean(axis=1)                                           # uniform mean
        features = keys * meta["scale"]                                    # s_j = a_j . q

        for head in range(meta["heads"]):
            centered = values[head] - mu[head]
            weighted = coefficients[head][:, None] * features[head]
            delta_m = weighted.T @ centered                                # -grad_B E(0)
            worst_identity = max(worst_identity,
                                 np.abs(direct[head] - (mu[head] + query[head] @ delta_m)).max())

            # dM is exactly minus the gradient at B = 0. Checked by directional
            # derivative along random directions: for a quadratic the central
            # difference is exact at any step, so a large step avoids the
            # cancellation a tiny one would suffer against a large E.
            hessian = weighted.T @ features[head]
            base = 0.5 * np.sum(coefficients[head][:, None] * centered ** 2)
            rng = np.random.default_rng(head)
            for _ in range(3):
                direction = rng.normal(size=delta_m.shape)
                direction /= np.linalg.norm(direction)
                step = 1e-2

                def energy(B):
                    err = features[head] @ B - centered
                    return 0.5 * np.sum(coefficients[head][:, None] * err ** 2)

                slope = (energy(step * direction) - energy(-step * direction)) / (2 * step)
                worst_gradient = max(worst_gradient,
                                     abs(slope + np.sum(delta_m * direction)))

            # does the unit step descend?
            residual = features[head] @ delta_m - centered
            stepped = 0.5 * np.sum(coefficients[head][:, None] * residual ** 2)
            increases += stepped > base
            total += 1
            lambdas.append(np.linalg.eigvalsh((hessian + hessian.T) / 2).max())

        # gauge freedom: any constant kappa reproduces y
        if context == 0:
            print("\ngauge freedom -- max |y_direct - y_reconstructed| per kappa")
            for kappa in (0.0, 1.0, 2.0, -3.0):
                gauge, _ = weights(scores, kappa)
                worst = 0.0
                for head in range(meta["heads"]):
                    centered = values[head] - mu[head]
                    dm = (gauge[head][:, None] * features[head]).T @ centered
                    worst = max(worst, np.abs(direct[head] - (mu[head] + query[head] @ dm)).max())
                label = "  the paper's choice" if kappa == 1.0 else ""
                print(f"  kappa = {kappa:>5} : {worst:.3e}{label}")

    print(f"\nidentity      max |y_direct - (mu + q.dM)| : {worst_identity:.3e}")
    print(f"gradient      max |<grad E(0) + dM, u>|      : {worst_gradient:.3e}")
    print(f"\ndescent       E(dM) > E(0) in               : {increases}/{total} heads "
          f"({100 * increases / total:.0f}%)")
    print(f"              lambda_max(H)  median {np.median(lambdas):.2f}  max {max(lambdas):.2f}")
    print("              a unit step descends only where lambda_max < 2.")
    return worst_identity


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", nargs="?", default="reference_capture.npz")
    raise SystemExit(0 if check(parser.parse_args().capture) < 1e-10 else 1)
