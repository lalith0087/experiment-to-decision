"""Is the heterogeneity real? Uncertainty and calibration for uplift scores.

Use out-of-fold scores (never scores from a model evaluated on its own training data).
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

from .evaluate import auuc


def _ols_hc1(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    XtXi = np.linalg.inv(X.T @ X)
    meat = (X * resid[:, None]).T @ (X * resid[:, None])
    n, k = X.shape
    cov = XtXi @ meat @ XtXi * n / (n - k)
    return beta, np.sqrt(np.diag(cov))


def calibration_test(y, t, score, e=None):
    """Best-linear-predictor test (Chernozhukov et al.) for randomized data.

    The IPW outcome w = (t - e) / (e (1 - e)) * y has E[w | x] = tau(x). Regress w on
    [1, score - mean(score)] with robust standard errors:
      * intercept  = estimated average treatment effect
      * slope      = how much of the predicted heterogeneity is real
    slope = 1 means well calibrated; slope = 0 means the scores carry no heterogeneity signal;
    slope > 1 means the model under-states differences, slope < 1 that it over-states them.
    Returns slope, se, 95% CI, and p-values for H0: slope = 0 and H0: slope = 1.
    """
    y, t, s = np.asarray(y, float), np.asarray(t).astype(float), np.asarray(score, float)
    e = t.mean() if e is None else e
    w = (t - e) / (e * (1 - e)) * y
    X = np.column_stack([np.ones(len(y)), s - s.mean()])
    b, se = _ols_hc1(X, w)
    z0, z1 = b[1] / se[1], (b[1] - 1) / se[1]
    return {"ate": float(b[0]), "ate_se": float(se[0]), "slope": float(b[1]), "slope_se": float(se[1]),
            "slope_ci": (float(b[1] - 1.96 * se[1]), float(b[1] + 1.96 * se[1])),
            "p_slope_zero": float(2 * (1 - norm.cdf(abs(z0)))),
            "p_slope_one": float(2 * (1 - norm.cdf(abs(z1))))}


def group_effects(y, t, score, groups=5):
    """Observed treatment effect per score group (group 1 = highest scores), with 95% CIs,
    plus the top-minus-bottom difference. A model with real signal has a clearly positive gap."""
    y, t = np.asarray(y, float), np.asarray(t).astype(bool)
    g = pd.qcut(pd.Series(-np.asarray(score)).rank(method="first"), groups, labels=False).values
    rows = []
    for k in range(groups):
        m = g == k
        yt, yc = y[m & t], y[m & ~t]
        d = yt.mean() - yc.mean()
        se = np.sqrt(yt.var(ddof=1) / len(yt) + yc.var(ddof=1) / len(yc))
        rows.append({"group": k + 1, "n": int(m.sum()), "effect": d, "se": se,
                     "ci_low": d - 1.96 * se, "ci_high": d + 1.96 * se})
    tbl = pd.DataFrame(rows)
    gap = tbl.effect.iloc[0] - tbl.effect.iloc[-1]
    gse = np.sqrt(tbl.se.iloc[0] ** 2 + tbl.se.iloc[-1] ** 2)
    return tbl, {"top_minus_bottom": float(gap), "se": float(gse),
                 "ci": (float(gap - 1.96 * gse), float(gap + 1.96 * gse)),
                 "p_value": float(2 * (1 - norm.cdf(abs(gap / gse))))}


def bootstrap_auuc(y, t, score, n_boot=200, seed=0):
    """Bootstrap interval for the Qini area. If the interval excludes 0 the ranking beats random."""
    y, t, s = np.asarray(y), np.asarray(t), np.asarray(score)
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        vals.append(auuc(y[i], t[i], s[i]))
    return {"auuc": float(auuc(y, t, s)), "ci": tuple(float(v) for v in np.percentile(vals, [2.5, 97.5]))}


def compare_auuc(y, t, score_a, score_b, n_boot=200, seed=0):
    """Paired bootstrap for the difference in Qini area between two models (a minus b)."""
    y, t, a, b = np.asarray(y), np.asarray(t), np.asarray(score_a), np.asarray(score_b)
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        diffs.append(auuc(y[i], t[i], a[i]) - auuc(y[i], t[i], b[i]))
    d = auuc(y, t, a) - auuc(y, t, b)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff": float(d), "ci": (float(lo), float(hi)), "distinguishable": bool(lo > 0 or hi < 0)}
