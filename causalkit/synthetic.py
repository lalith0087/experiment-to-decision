"""Synthetic randomized data with a known ground-truth CATE."""
import numpy as np
import pandas as pd


def make_uplift_data(n=20000, n_features=5, treat_prob=0.5, noise=1.0, seed=0):
    """Outcome y = baseline(x) + tau(x) * t + noise, with heterogeneous tau.

    tau(x) = 1.0*x0 + 0.5*(x1 > 0), so some users benefit, some are hurt, and
    x2..x4 are pure noise features. Returns (DataFrame, true_tau).
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features))
    t = rng.binomial(1, treat_prob, n)
    tau = 1.0 * X[:, 0] + 0.5 * (X[:, 1] > 0)
    base = 2.0 * X[:, 2] + X[:, 3] ** 2
    y = base + tau * t + rng.normal(0, noise, n)
    # pre-experiment metric correlated with the baseline (for CUPED)
    pre = base + rng.normal(0, 1.0, n)
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(n_features)])
    df["t"], df["y"], df["pre"] = t, y, pre
    return df, tau
