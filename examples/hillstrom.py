"""Real-data analysis: Hillstrom MineThatData email experiment (64k customers, randomized).
Treatment = any email (men's or women's) vs no email. Run: python examples/hillstrom.py"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from causalkit import *

d = pd.read_csv("data/hillstrom.csv")
d["t"] = (d.segment != "No E-Mail").astype(int)
X = pd.get_dummies(d[["recency", "history", "mens", "womens", "newbie", "zip_code", "channel"]],
                   drop_first=True).astype(float)

print("== SRM (3 arms) ==", srm_test(d.segment.value_counts().values))
for y in ("visit", "conversion", "spend"):
    n = naive_ate(d[y], d.t); c = cuped_ate(d[y], d.t, d.history)
    print(f"{y:10s} naive ate={n['ate']:.4f} se={n['se']:.4f} p={n['p_value']:.1e} | "
          f"cuped se={c['se']:.4f} varred={100*c['variance_reduction']:.1f}%")
print("Per arm vs control (visit):")
ctrl = d[d.segment == "No E-Mail"]
for s in ("Mens E-Mail", "Womens E-Mail"):
    a = d[d.segment == s]; r = naive_ate(pd.concat([a, ctrl]).visit, (pd.concat([a, ctrl]).segment == s))
    print(f"  {s:14s} lift={r['ate']:.4f} ci=({r['ci'][0]:.4f},{r['ci'][1]:.4f})")

print("\n== Uplift models, 5-fold CV, outcome = visit ==")
y = d.visit.values; t = d.t.values
skf = StratifiedKFold(5, shuffle=True, random_state=0)
oof = {k: np.zeros(len(d)) for k in ("T", "X", "DR")}
scores = {k: [] for k in oof}; rnd = []
rng = np.random.default_rng(0)
for tr, te in skf.split(X, t * 2 + y):
    for k, cls in (("T", TLearner), ("X", XLearner), ("DR", DRLearner)):
        p = cls().fit(X.iloc[tr], t[tr], y[tr]).predict(X.iloc[te])
        oof[k][te] = p; scores[k].append(auuc(y[te], t[te], p))
    rnd.append(auuc(y[te], t[te], rng.normal(size=len(te))))
for k in oof: print(f"{k:3s} auuc mean={np.mean(scores[k]):.5f} sd={np.std(scores[k]):.5f}")
print(f"random auuc mean={np.mean(rnd):.5f} sd={np.std(rnd):.5f}")

best = max(oof, key=lambda k: np.mean(scores[k]))
print(f"\nUplift by decile ({best}, out-of-fold):")
print(uplift_by_decile(y, t, oof[best]).round(4).to_string(index=False))

plt.figure(figsize=(6, 4))
for k in oof:
    f, g, _ = qini_curve(y, t, oof[k]); plt.plot(f, g, label=k)
plt.plot(f, _, "k--", label="random"); plt.xlabel("fraction emailed"); plt.ylabel("incremental visits (per customer)")
plt.legend(); plt.tight_layout(); plt.savefig("docs/hillstrom_qini.png", dpi=120)
