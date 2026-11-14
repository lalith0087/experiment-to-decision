"""End-to-end demo on synthetic data with known ground truth. Run: python examples/demo.py"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from causalkit import *

df, tau = make_uplift_data(n=30000, seed=7)
print("== 1. Experiment health ==")
print(srm_test(np.bincount(df.t)))

print("\n== 2. CUPED vs naive ATE (true ATE = %.3f) ==" % tau.mean())
for name, r in [("naive", naive_ate(df.y, df.t)), ("cuped", cuped_ate(df.y, df.t, df.pre))]:
    print(f"{name:6s} ate={r['ate']:.3f} se={r['se']:.4f} ci=({r['ci'][0]:.3f},{r['ci'][1]:.3f})")
print("variance reduction: %.1f%%" % (100 * cuped_ate(df.y, df.t, df.pre)["variance_reduction"]))

print("\n== 3. Sequential test (peek-safe) ==")
r = msprt(df.y.values, df.t.values, tau=0.3)
print(f"stopped at n={r['stopped_at']} of {len(df)}, always-valid p={r['final_p']:.2e}")

print("\n== 4. Uplift models on 10k holdout (corr with true tau, Qini area) ==")
feats = [c for c in df.columns if c.startswith("x")]
tr, te = df.index < 20000, df.index >= 20000
plt.figure(figsize=(6, 4))
for cls in (TLearner, XLearner, DRLearner):
    m = cls().fit(df.loc[tr, feats], df.t[tr], df.y[tr])
    p = m.predict(df.loc[te, feats])
    print(f"{cls.__name__:10s} corr={np.corrcoef(p, tau[te])[0,1]:.3f} auuc={auuc(df.y[te], df.t[te], p):.4f}")
    f, g, _ = qini_curve(df.y[te], df.t[te], p)
    plt.plot(f, g, label=cls.__name__)
f, _, rnd = qini_curve(df.y[te], df.t[te], p)
plt.plot(f, rnd, "k--", label="random")
plt.xlabel("fraction targeted"); plt.ylabel("incremental gain"); plt.legend(); plt.tight_layout()
plt.savefig("docs/qini.png", dpi=120)
print("\nsaved docs/qini.png")
