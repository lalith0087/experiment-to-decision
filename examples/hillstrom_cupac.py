"""Can an ML covariate rescue variance reduction on Hillstrom? Naive vs CUPED vs linear-all vs CUPAC."""
import numpy as np, pandas as pd
from causalkit import naive_ate, cuped_ate, cupac_ate

d = pd.read_csv("data/hillstrom.csv")
t = (d.segment != "No E-Mail").astype(int).values
X = pd.get_dummies(d[["recency", "history", "mens", "womens", "newbie", "zip_code", "channel"]],
                   drop_first=True).astype(float).values
rows = []
for y in ("visit", "conversion", "spend"):
    v = d[y].values.astype(float)
    base = naive_ate(v, t)
    for name, r in (("naive", base), ("CUPED (history)", cuped_ate(v, t, d.history.values)),
                    ("CUPED (all 8, linear)", cuped_ate(v, t, X)), ("CUPAC (boosted trees)", cupac_ate(v, t, X))):
        rows.append({"outcome": y, "method": name, "ate": r["ate"], "se": r["se"],
                     "se_vs_naive": 1 - r["se"] / base["se"],
                     "var_reduction": r.get("variance_reduction", 0.0), "corr": r.get("covariate_corr", np.nan)})
print(pd.DataFrame(rows).round(4).to_string(index=False))
