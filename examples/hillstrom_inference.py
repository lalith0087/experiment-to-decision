"""Is the heterogeneity on Hillstrom real? Calibration, group effects and Qini intervals."""
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from causalkit import *

d = pd.read_csv("data/hillstrom.csv")
d["t"] = (d.segment != "No E-Mail").astype(int)
X = pd.get_dummies(d[["recency", "history", "mens", "womens", "newbie", "zip_code", "channel"]],
                   drop_first=True).astype(float)
t = d.t.values

def oof(cls, target):
    out = np.zeros(len(d))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, t * 2 + d.visit.values):
        out[te] = cls().fit(X.iloc[tr], t[tr], target[tr]).predict(X.iloc[te])
    return out

for outcome in ("visit", "spend"):
    y = d[outcome].values.astype(float)
    S = {"T": oof(TLearner, y), "X": oof(XLearner, y), "DR": oof(DRLearner, y)}
    print(f"\n================ outcome = {outcome} ================")
    for k, s in S.items():
        c = calibration_test(y, t, s)
        b = bootstrap_auuc(y, t, s, n_boot=200)
        _, gap = group_effects(y, t, s)
        print(f"{k:2s} calibration slope={c['slope']:.2f} CI=({c['slope_ci'][0]:.2f},{c['slope_ci'][1]:.2f}) "
              f"p(slope=0)={c['p_slope_zero']:.3f} | top-minus-bottom quintile={gap['top_minus_bottom']:.4f} "
              f"CI=({gap['ci'][0]:.4f},{gap['ci'][1]:.4f}) | AUUC={b['auuc']:.5f} CI=({b['ci'][0]:.5f},{b['ci'][1]:.5f})")
    for a, b2 in (("T", "X"), ("T", "DR"), ("X", "DR")):
        r = compare_auuc(y, t, S[a], S[b2], n_boot=200)
        print(f"  {a} minus {b2}: diff={r['diff']:.5f} CI=({r['ci'][0]:.5f},{r['ci'][1]:.5f}) distinguishable={r['distinguishable']}")
    tbl, _ = group_effects(y, t, S["DR"])
    print("DR quintile effects:"); print(tbl.round(4).to_string(index=False))
