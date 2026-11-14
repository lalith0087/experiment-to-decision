"""Who should we email? Profit-based targeting on Hillstrom, with honest uncertainty.
Value = spend (treat as contribution margin); cost = per-email cost, swept below."""
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from causalkit import *

d = pd.read_csv("data/hillstrom.csv")
d["t"] = (d.segment != "No E-Mail").astype(int)
X = pd.get_dummies(d[["recency", "history", "mens", "womens", "newbie", "zip_code", "channel"]],
                   drop_first=True).astype(float)
y, t, visit = d.spend.values, d.t.values, d.visit.values

def oof(cls, target):
    out = np.zeros(len(d))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, t * 2 + visit):
        out[te] = cls().fit(X.iloc[tr], t[tr], target[tr]).predict(X.iloc[te])
    return out

scorers = {"DR on spend": oof(DRLearner, y), "X on spend": oof(XLearner, y),
           "DR on visit": oof(DRLearner, visit.astype(float))}
rows = []
for cost in (0.1, 0.3, 0.5, 0.7):
    for name, s in scorers.items():
        r = evaluate_policy(y, t, s, cost=cost, n_boot=300)
        rows.append({"cost": cost, "scorer": name, "treat_all": r["treat_all_gain"],
                     "policy": r["policy_gain"], "policy_ci": "(%.3f, %.3f)" % r["policy_ci"],
                     "share": r["chosen_share"], "policy-all": r["policy_minus_all"],
                     "diff_ci": "(%.3f, %.3f)" % r["policy_minus_all_ci"]})
print(pd.DataFrame(rows).round(3).to_string(index=False))
