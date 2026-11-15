"""Renders docs/peeking.gif: why peeking breaks a naive test and how mSPRT fixes it.
Both tests watch the SAME stream of an experiment with NO true effect. Run: python examples/make_gif.py"""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.stats import norm
from causalkit import msprt

rng = np.random.default_rng(11)
n = 6000
while True:  # pick a seed-stream where the naive test is fooled (typical under peeking)
    t = rng.integers(0, 2, n); y = rng.normal(size=n)
    nt, nc = np.cumsum(t), np.cumsum(1 - t)
    mt, mc = np.cumsum(y * t) / np.maximum(nt, 1), np.cumsum(y * (1 - t)) / np.maximum(nc, 1)
    z = (mt - mc) / np.sqrt(1 / np.maximum(nt, 1) + 1 / np.maximum(nc, 1))
    naive = 2 * (1 - norm.cdf(np.abs(z)))
    naive[:60] = 1
    if (naive[200:] < 0.05).any() and msprt(y, t, tau=0.2)["stopped_at"] is None:
        break
seq = msprt(y, t, tau=0.2)["p_values"]

fig, ax = plt.subplots(figsize=(7, 4))
steps = np.linspace(60, n, 45).astype(int)
def draw(k):
    ax.clear(); i = steps[min(k, len(steps) - 1)]; x = np.arange(i)
    ax.plot(x, naive[:i], color="#d95f02", lw=1.2, label="naive p-value (peeking)")
    ax.plot(x, seq[:i], color="#1b9e77", lw=2, label="mSPRT always-valid p-value")
    ax.axhline(0.05, color="k", ls="--", lw=1)
    ax.set_xlim(0, n); ax.set_ylim(0, 1); ax.set_xlabel("customers observed"); ax.set_ylabel("p-value")
    fooled = (naive[:i] < 0.05).any()
    ax.set_title("No true effect: peeking says " + ("'significant!'" if fooled else "'not yet'")
                 + "; mSPRT stays valid", fontsize=10, color="#d95f02" if fooled else "k")
    ax.legend(loc="upper right", fontsize=8)
    if fooled:
        j = np.argmax(naive[:i] < 0.05); ax.plot(j, naive[j], "o", color="#d95f02")
FuncAnimation(fig, draw, frames=len(steps) + 8, interval=120).save(
    "docs/peeking.gif", writer=PillowWriter(fps=8), dpi=80)
