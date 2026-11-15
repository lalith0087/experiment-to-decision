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


def cupac_ate(y, t, X, model=None, n_splits=5, seed=0, alpha=0.05):
    """CUPAC: CUPED with an ML-predicted covariate built from pre-experiment features `X`.

    A regressor learns E[y | X] on control users only; predictions are cross-fitted so no
    user's covariate comes from a model that saw their own outcome. Because the prediction is a
    function of pre-treatment features only, it is independent of assignment and the ATE
    estimate stays unbiased. It reduces variance only to the extent X predicts y non-linearly
    or in combination (a strong linear covariate is already handled by cuped_ate).
    """
    from sklearn.base import clone
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import KFold

    y, t = np.asarray(y, float), np.asarray(t).astype(bool)
    X = np.asarray(X, float)
    model = model or HistGradientBoostingRegressor(max_depth=3, max_iter=100, learning_rate=0.05)
    pred = np.zeros(len(y))
    for tr, te in KFold(n_splits, shuffle=True, random_state=seed).split(X):
        ctrl = tr[~t[tr]]
        pred[te] = clone(model).fit(X[ctrl], y[ctrl]).predict(X[te])
    out = cuped_ate(y, t, pred, alpha)
    out["covariate_corr"] = float(np.corrcoef(pred, y)[0, 1])
    return out
