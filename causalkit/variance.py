"""Average treatment effect estimators, with CUPED variance reduction."""
import numpy as np
from scipy.stats import norm


def _result(ate, se, alpha):
    z = norm.ppf(1 - alpha / 2)
    return {"ate": float(ate), "se": float(se), "ci": (float(ate - z * se), float(ate + z * se)),
            "p_value": float(2 * (1 - norm.cdf(abs(ate / se))))}


def naive_ate(y, t, alpha=0.05):
    y, t = np.asarray(y, float), np.asarray(t).astype(bool)
    yt, yc = y[t], y[~t]
    se = np.sqrt(yt.var(ddof=1) / len(yt) + yc.var(ddof=1) / len(yc))
    return _result(yt.mean() - yc.mean(), se, alpha)


def cuped_ate(y, t, x, alpha=0.05):
    """CUPED: adjust outcome by a pre-experiment covariate `x`.

    theta = cov(y, x) / var(x) is estimated on pooled data (valid because x is
    pre-treatment, so it is independent of assignment). Accepts a 1-D or 2-D x.
    Adds `variance_reduction` = 1 - var(adjusted) / var(raw).
    """
    y, t = np.asarray(y, float), np.asarray(t).astype(bool)
    x = np.asarray(x, float)
    if x.ndim == 1:
        x = x[:, None]
    xc = x - x.mean(0)
    theta, *_ = np.linalg.lstsq(xc, y - y.mean(), rcond=None)
    y_adj = y - xc @ theta
    out = naive_ate(y_adj, t, alpha)
    out["variance_reduction"] = float(1 - y_adj.var() / y.var())
    return out
