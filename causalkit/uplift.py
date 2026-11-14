"""Meta-learners for heterogeneous treatment effects (CATE).

All learners share the interface fit(X, t, y) -> self, predict(X) -> tau_hat.
Designed for randomized experiments: propensity is the observed treated share.
"""
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold


def _default():
    return HistGradientBoostingRegressor(max_depth=3, max_iter=150, learning_rate=0.05)


class TLearner:
    """Fit separate outcome models per arm; tau = mu1(x) - mu0(x)."""

    def __init__(self, model=None):
        self.model = model or _default()

    def fit(self, X, t, y):
        X, t, y = np.asarray(X), np.asarray(t).astype(bool), np.asarray(y)
        self.m1_ = clone(self.model).fit(X[t], y[t])
        self.m0_ = clone(self.model).fit(X[~t], y[~t])
        return self

    def predict(self, X):
        X = np.asarray(X)
        return self.m1_.predict(X) - self.m0_.predict(X)


class XLearner:
    """Kunzel et al. X-learner: impute individual effects, then blend by propensity."""

    def __init__(self, model=None):
        self.model = model or _default()

    def fit(self, X, t, y):
        X, t, y = np.asarray(X), np.asarray(t).astype(bool), np.asarray(y)
        m1 = clone(self.model).fit(X[t], y[t])
        m0 = clone(self.model).fit(X[~t], y[~t])
        d1 = y[t] - m0.predict(X[t])      # treated: observed - imputed control
        d0 = m1.predict(X[~t]) - y[~t]    # control: imputed treated - observed
        self.tau1_ = clone(self.model).fit(X[t], d1)
        self.tau0_ = clone(self.model).fit(X[~t], d0)
        self.e_ = t.mean()
        return self

    def predict(self, X):
        X = np.asarray(X)
        return self.e_ * self.tau0_.predict(X) + (1 - self.e_) * self.tau1_.predict(X)


class DRLearner:
    """Doubly-robust learner with cross-fitted nuisances and a final-stage regressor
    on the AIPW pseudo-outcome. Unbiased pseudo-outcome even if outcome models are off."""

    def __init__(self, model=None, final_model=None, n_splits=5, seed=0):
        self.model = model or _default()
        self.final_model = final_model or _default()
        self.n_splits, self.seed = n_splits, seed

    def fit(self, X, t, y):
        X, t, y = np.asarray(X), np.asarray(t).astype(bool), np.asarray(y)
        e = t.mean()
        mu1, mu0 = np.zeros(len(y)), np.zeros(len(y))
        for tr, te in KFold(self.n_splits, shuffle=True, random_state=self.seed).split(X):
            a, b = tr[t[tr]], tr[~t[tr]]
            mu1[te] = clone(self.model).fit(X[a], y[a]).predict(X[te])
            mu0[te] = clone(self.model).fit(X[b], y[b]).predict(X[te])
        pseudo = mu1 - mu0 + t * (y - mu1) / e - (~t) * (y - mu0) / (1 - e)
        self.final_ = clone(self.final_model).fit(X, pseudo)
        return self

    def predict(self, X):
        return self.final_.predict(np.asarray(X))
