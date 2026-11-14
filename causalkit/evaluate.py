"""Evaluating uplift models on randomized holdout data (ground truth not needed)."""
import numpy as np
import pandas as pd

_trapz = getattr(np, "trapezoid", None) or np.trapz


def qini_curve(y, t, score):
    """Cumulative incremental-gain curve when targeting users in descending `score` order.

    x = fraction of population targeted; gain(x) = (mean_y_treated - mean_y_control among
    the top-x) * x. A random policy yields the straight line x * ATE.
    Returns (fractions, gains, random_baseline).
    """
    y, t, score = np.asarray(y, float), np.asarray(t).astype(bool), np.asarray(score)
    order = np.argsort(-score, kind="stable")
    y, t = y[order], t[order]
    n = len(y)
    nt, nc = np.cumsum(t), np.cumsum(~t)
    st, sc = np.cumsum(y * t), np.cumsum(y * ~t)
    with np.errstate(divide="ignore", invalid="ignore"):
        diff = np.where((nt > 0) & (nc > 0), st / nt - sc / nc, 0.0)
    frac = np.arange(1, n + 1) / n
    gain = diff * frac
    ate = diff[-1]
    return np.r_[0, frac], np.r_[0, gain], np.r_[0, frac * ate]


def auuc(y, t, score):
    """Area between the model's gain curve and the random baseline (Qini coefficient).

    Positive => the ranking beats random targeting. Compare across models on the same
    holdout; it is not comparable across datasets.
    """
    f, g, r = qini_curve(y, t, score)
    return float(_trapz(g - r, f))


def uplift_by_decile(y, t, score, bins=10):
    """Observed treatment effect per score decile (decile 1 = highest predicted uplift).

    A good model shows a monotonically decreasing observed-uplift column.
    """
    df = pd.DataFrame({"y": y, "t": np.asarray(t).astype(bool), "s": score})
    df["decile"] = pd.qcut(-df["s"].rank(method="first"), bins, labels=range(1, bins + 1))
    rows = []
    for d, g in df.groupby("decile", observed=True):
        rows.append({"decile": int(d), "n": len(g), "pred_uplift": g["s"].mean(),
                     "obs_uplift": g.loc[g.t, "y"].mean() - g.loc[~g.t, "y"].mean()})
    return pd.DataFrame(rows)
