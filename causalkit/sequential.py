"""Always-valid inference via the mixture sequential probability ratio test."""
import numpy as np


def msprt(y, t, tau=None, alpha=0.05):
    """mSPRT for a difference in means with a N(0, tau^2) mixing prior.

    Streams through observations in the given order and returns the running
    always-valid p-value, so peeking at any time does not inflate Type I error.
    Variance is estimated from the data seen so far (plug-in), which is accurate
    once each arm has a few dozen observations.

    Returns dict: p_values (array), stopped_at (index or None), final_p.
    """
    y, t = np.asarray(y, float), np.asarray(t).astype(bool)
    n = len(y)
    nt = np.cumsum(t)
    nc = np.cumsum(~t)
    st, sc = np.cumsum(y * t), np.cumsum(y * ~t)
    sst, ssc = np.cumsum(y**2 * t), np.cumsum(y**2 * ~t)

    p = np.ones(n)
    running = 1.0
    ok = (nt >= 30) & (nc >= 30)
    for i in range(n):
        if not ok[i]:
            p[i] = running
            continue
        mt, mc = st[i] / nt[i], sc[i] / nc[i]
        vt = max(sst[i] / nt[i] - mt**2, 1e-12)
        vc = max(ssc[i] / nc[i] - mc**2, 1e-12)
        V = vt / nt[i] + vc / nc[i]
        theta = mt - mc
        tau2 = (tau**2) if tau is not None else (np.sqrt(vt + vc) * 0.1) ** 2
        lam = np.sqrt(V / (V + tau2)) * np.exp(tau2 * theta**2 / (2 * V * (V + tau2)))
        running = min(running, 1.0 / lam)
        p[i] = running
    hits = np.where(p < alpha)[0]
    return {"p_values": p, "stopped_at": int(hits[0]) if len(hits) else None, "final_p": float(p[-1])}
