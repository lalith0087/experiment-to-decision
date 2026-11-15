"""Interactive demo of experiment-to-decision. Run: streamlit run app/streamlit_app.py"""
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import norm
from sklearn.model_selection import KFold

from causalkit import (DRLearner, auuc, cuped_ate, evaluate_policy, make_uplift_data, mde_mean,
                       msprt, naive_ate, policy_curve, qini_curve, sample_size_mean,
                       sample_size_proportion, srm_test, uplift_by_decile)

st.set_page_config(page_title="experiment-to-decision", layout="wide")
st.title("experiment-to-decision")
st.caption("Design, check, analyze and target an A/B test. All data here is simulated in your session "
           "with a known ground truth, so you can see when each method works and when it does not.")

t_power, t_cuped, t_peek, t_target = st.tabs(["1. Power", "2. CUPED", "3. Peeking", "4. Targeting"])

# ---------------------------------------------------------------- 1. power
with t_power:
    st.subheader("How many users do I need?")
    kind = st.radio("Metric", ["Conversion rate", "Continuous (e.g. revenue)"], horizontal=True)
    c1, c2, c3 = st.columns(3)
    alpha = c1.select_slider("Significance level", [0.01, 0.05, 0.10], value=0.05)
    power = c2.select_slider("Power", [0.7, 0.8, 0.9, 0.95], value=0.8)
    if kind == "Conversion rate":
        base = c3.slider("Baseline rate", 0.01, 0.5, 0.10, 0.01)
        mde = st.slider("Smallest lift worth detecting (absolute)", 0.002, 0.10, 0.02, 0.002)
        n_c, n_t = sample_size_proportion(base, mde, alpha, power)
        curve = pd.DataFrame({"lift": (ls := np.linspace(0.005, 0.10, 40)),
                              "users per arm": [sample_size_proportion(base, m, alpha, power)[0] for m in ls]})
    else:
        sd = c3.number_input("Standard deviation of the metric", 0.1, 1000.0, 20.0)
        mde = st.slider("Smallest effect worth detecting", 0.05, 10.0, 1.0, 0.05)
        n_c, n_t = sample_size_mean(mde, sd, alpha, power)
        curve = pd.DataFrame({"lift": (ls := np.linspace(0.1, 10, 40)),
                              "users per arm": [sample_size_mean(m, sd, alpha, power)[0] for m in ls]})
    st.metric("Users needed per arm", f"{n_c:,}", help="Two-sided z-test, equal split.")
    st.line_chart(curve, x="lift", y="users per arm")
    st.caption("Halving the effect you want to detect roughly quadruples the sample you need.")

# ---------------------------------------------------------------- 2. CUPED
with t_cuped:
    st.subheader("CUPED: use pre-experiment behaviour to cut noise")
    c1, c2, c3 = st.columns(3)
    n = c1.slider("Users", 2000, 40000, 10000, 1000, key="cn")
    rho = c2.slider("Correlation of pre-period metric with outcome", 0.0, 0.95, 0.6, 0.05)
    true_eff = c3.slider("True effect", 0.0, 0.5, 0.1, 0.05)
    seed = st.number_input("Random seed", 0, 9999, 1)
    rng = np.random.default_rng(int(seed))
    pre = rng.normal(size=n)
    y = rho * pre + np.sqrt(1 - rho**2) * rng.normal(size=n)
    t = rng.integers(0, 2, n)
    y = y + true_eff * t
    a, c = naive_ate(y, t), cuped_ate(y, t, pre)
    d = pd.DataFrame({"estimate": [a["ate"], c["ate"]], "std error": [a["se"], c["se"]],
                      "CI low": [a["ci"][0], c["ci"][0]], "CI high": [a["ci"][1], c["ci"][1]],
                      "p-value": [a["p_value"], c["p_value"]]}, index=["naive", "CUPED"])
    st.dataframe(d.style.format("{:.4f}"))
    st.metric("Variance reduction", f"{100 * c['variance_reduction']:.0f}%",
              help="Theory: about rho squared. Try rho = 0 to see it vanish.")
    st.caption("On the real Hillstrom email experiment, purchase history barely predicted visits, "
               "so CUPED removed only 0.4% of variance. It only helps with a strong pre-period signal.")

# ---------------------------------------------------------------- 3. peeking
with t_peek:
    st.subheader("Peeking: check a test every day and you will 'find' effects that are not there")
    c1, c2, c3 = st.columns(3)
    n_obs = c1.slider("Users per experiment", 500, 4000, 2000, 500)
    every = c2.select_slider("Check the p-value every N users", [1, 10, 50, 200, 1000], value=10)
    reps = c3.slider("Simulated experiments (no true effect)", 50, 300, 150, 50)
    if st.button("Run simulation", type="primary"):
        rng = np.random.default_rng(0)
        fn = fs = 0
        bar = st.progress(0.0)
        for r in range(reps):
            t = rng.integers(0, 2, n_obs)
            y = rng.normal(size=n_obs)
            nt, nc = np.cumsum(t), np.cumsum(1 - t)
            ok = (nt >= 30) & (nc >= 30)
            mt = np.cumsum(y * t) / np.maximum(nt, 1)
            mc = np.cumsum(y * (1 - t)) / np.maximum(nc, 1)
            z = (mt - mc) / np.sqrt(1 / np.maximum(nt, 1) + 1 / np.maximum(nc, 1))
            p = 2 * (1 - norm.cdf(np.abs(z)))
            idx = np.arange(every - 1, n_obs, every)
            fn += bool((p[idx][ok[idx]] < 0.05).any())
            fs += msprt(y, t, tau=0.2)["stopped_at"] is not None
            bar.progress((r + 1) / reps)
        c1, c2 = st.columns(2)
        c1.metric("Naive test: false positives", f"{100 * fn / reps:.0f}%", help="Nominal rate is 5%.")
        c2.metric("mSPRT: false positives", f"{100 * fs / reps:.0f}%")
        st.caption(f"{reps} experiments, no real effect, p-value checked every {every} users. "
                   "A valid test should stay near 5% or below. mSPRT keeps its error rate however often you look.")

# ---------------------------------------------------------------- 4. targeting
@st.cache_data(show_spinner="Training uplift models on a simulated experiment...")
def _fit(n, seed):
    df, tau = make_uplift_data(n=n, seed=seed)
    feats = [c for c in df.columns if c.startswith("x")]
    X = df[feats].values
    oof = np.zeros(n)
    for tr, te in KFold(2, shuffle=True, random_state=0).split(X):
        oof[te] = DRLearner().fit(X[tr], df.t.values[tr], df.y.values[tr]).predict(X[te])
    return df, tau, oof


with t_target:
    st.subheader("Who should we treat? (and is the answer trustworthy?)")
    c1, c2, c3 = st.columns(3)
    n = c1.slider("Users", 4000, 20000, 10000, 2000, key="tn")
    seed = c2.number_input("Random seed", 0, 9999, 7, key="ts")
    cost = c3.slider("Cost of treating one user", 0.0, 1.5, 0.5, 0.05)
    df, tau, oof = _fit(n, int(seed))
    st.caption("Simulated randomized experiment where the true effect varies by user "
               f"(true average effect {tau.mean():.2f}). Scores are out-of-fold from a doubly-robust learner.")
    q1, q2 = st.columns(2)
    f, g, r = qini_curve(df.y, df.t, oof)
    q1.markdown("**Qini curve: gain from targeting by score**")
    q1.line_chart(pd.DataFrame({"fraction targeted": f[::20], "model": g[::20], "random": r[::20]}),
                  x="fraction targeted")
    q1.caption(f"Qini area: {auuc(df.y, df.t, oof):.3f} (random is about 0).")
    q2.markdown("**Observed uplift by predicted decile** (should fall left to right)")
    q2.bar_chart(uplift_by_decile(df.y, df.t, oof), x="decile", y="obs_uplift")
    pc = policy_curve(df.y, df.t, oof, cost=cost)
    st.markdown("**Expected profit per user vs. share treated**")
    st.line_chart(pc, x="frac", y="gain")
    res = evaluate_policy(df.y, df.t, oof, cost=cost, n_boot=150)
    m1, m2, m3 = st.columns(3)
    m1.metric("Treat everyone", f"{res['treat_all_gain']:.3f}",
              help=f"95% CI {res['treat_all_ci'][0]:.3f} to {res['treat_all_ci'][1]:.3f}")
    m2.metric("Targeted policy (out-of-sample)", f"{res['policy_gain']:.3f}",
              help=f"95% CI {res['policy_ci'][0]:.3f} to {res['policy_ci'][1]:.3f}")
    lo, hi = res["policy_minus_all_ci"]
    m3.metric("Targeted minus treat-everyone", f"{res['policy_minus_all']:.3f}")
    verdict = ("Targeting beats treating everyone (interval excludes 0)." if lo > 0 else
               "Targeting is worse than treating everyone (interval excludes 0)." if hi < 0 else
               "No provable difference: the interval includes 0.")
    st.info(f"{verdict}  95% interval for the difference: {lo:.3f} to {hi:.3f}. "
            f"Chosen share to treat: {100 * res['chosen_share']:.0f}%.")
    st.caption("The share to treat is chosen on one half of the data and measured on the other, "
               "then the whole procedure is bootstrapped. On the real Hillstrom email experiment, "
               "targeting did not provably beat emailing everyone; here the effect is built to vary, "
               "so you can see what a win looks like.")
