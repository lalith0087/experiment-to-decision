"""Turning uplift scores into a targeting decision with honest profit estimates.

Everything uses inverse-propensity weighting on randomized data. For a customer i,
    w_i = t_i * (y_i - cost) / e  -  (1 - t_i) * y_i / (1 - e)
is an unbiased estimate of "profit if treated minus profit if not treated". The expected
gain of treating a set S per customer in the population is mean(w_i * 1[i in S]).
"""
import numpy as np
import pandas as pd


def _weights(y, t, cost, e):
    y, t = np.asarray(y, float), np.asarray(t).astype(bool)
    e = t.mean() if e is None else e
    return np.where(t, (y - cost) / e, -y / (1 - e))


def policy_curve(y, t, score, cost=0.0, fracs=None, e=None):
    """Expected per-customer gain vs. treating no one, as a function of the share treated
    (top share by descending `score`). frac=1.0 is 'treat everyone'."""
    w = _weights(y, t, cost, e)
    order = np.argsort(-np.asarray(score), kind="stable")
    cum = np.cumsum(w[order]) / len(w)
    fracs = np.linspace(0.05, 1.0, 20) if fracs is None else np.asarray(fracs)
    idx = np.clip(np.ceil(fracs * len(w)).astype(int) - 1, 0, len(w) - 1)
    return pd.DataFrame({"frac": fracs, "gain": cum[idx]})


def _select_then_eval(w, score, rng, fracs):
    """Choose the best treated share on one random half, evaluate it on the other (and swap).
    Selecting and scoring on the same data would overstate the gain."""
    n = len(w)
    perm = rng.permutation(n)
    halves = (perm[: n // 2], perm[n // 2:])
    out = []
    for a, b in ((0, 1), (1, 0)):
        ia, ib = halves[a], halves[b]
        oa, ob = ia[np.argsort(-score[ia], kind="stable")], ib[np.argsort(-score[ib], kind="stable")]
        ca, cb = np.cumsum(w[oa]) / len(ia), np.cumsum(w[ob]) / len(ib)
        ka = np.clip(np.ceil(fracs * len(ia)).astype(int) - 1, 0, len(ia) - 1)
        best = int(np.argmax(ca[ka]))
        kb = int(np.clip(np.ceil(fracs[best] * len(ib)) - 1, 0, len(ib) - 1))
        out.append((cb[kb], fracs[best]))
    return np.mean([o[0] for o in out]), np.mean([o[1] for o in out])


def evaluate_policy(y, t, score, cost=0.0, n_boot=300, seed=0, e=None):
    """Bootstrap the full 'pick a targeting share, then measure it out-of-sample' procedure.

    Returns a dict with per-customer gain vs. no-treatment for: the learned policy,
    treat-everyone, and the chosen share, each with a 95% bootstrap interval.
    """
    y, t, score = np.asarray(y, float), np.asarray(t).astype(bool), np.asarray(score, float)
    w = _weights(y, t, cost, e)
    fracs = np.linspace(0.05, 1.0, 20)
    rng = np.random.default_rng(seed)
    pol, share, allg = [], [], []
    n = len(w)
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        g, k = _select_then_eval(w[i], score[i], rng, fracs)
        pol.append(g); share.append(k); allg.append(w[i].mean())
    q = lambda v: tuple(np.percentile(v, [2.5, 97.5]))
    return {"policy_gain": float(np.mean(pol)), "policy_ci": q(pol),
            "treat_all_gain": float(w.mean()), "treat_all_ci": q(allg),
            "chosen_share": float(np.mean(share)),
            "policy_minus_all": float(np.mean(np.array(pol) - np.array(allg))),
            "policy_minus_all_ci": q(np.array(pol) - np.array(allg))}
