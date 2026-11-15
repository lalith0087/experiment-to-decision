import numpy as np
import pytest
from causalkit import (sample_size_mean, sample_size_proportion, mde_mean, srm_test,
                       naive_ate, cuped_ate, msprt, TLearner, XLearner, DRLearner,
                       qini_curve, auuc, uplift_by_decile, make_uplift_data)


def test_sample_size_known_value():
    # classic: sd=1, mde=0.2, alpha=.05, power=.8 -> ~393 per arm
    n_c, n_t = sample_size_mean(0.2, 1.0)
    assert 390 <= n_c <= 395 and n_c == n_t


def test_mde_inverts_sample_size():
    n_c, _ = sample_size_mean(0.2, 1.0)
    assert mde_mean(n_c, 1.0) == pytest.approx(0.2, rel=0.01)


def test_proportion_sample_size_reasonable():
    n_c, _ = sample_size_proportion(0.10, 0.02)
    assert 3000 < n_c < 4500


def test_srm_detects_mismatch():
    assert srm_test([5000, 5000])["srm"] is False
    assert srm_test([5000, 5400])["srm"] is True


def test_cuped_reduces_variance_and_stays_unbiased():
    df, tau = make_uplift_data(n=40000, seed=1)
    a, c = naive_ate(df.y, df.t), cuped_ate(df.y, df.t, df.pre)
    assert c["se"] < a["se"]
    assert c["variance_reduction"] > 0.2
    assert abs(c["ate"] - tau.mean()) < 4 * c["se"]


def test_cuped_ci_coverage():
    cover, reps = 0, 200
    for s in range(reps):
        df, tau = make_uplift_data(n=2000, seed=100 + s)
        lo, hi = cuped_ate(df.y, df.t, df.pre)["ci"]
        cover += lo <= tau.mean() <= hi
    assert 0.90 <= cover / reps <= 0.99


def test_msprt_controls_false_positives_under_peeking():
    rng = np.random.default_rng(0)
    fp, reps = 0, 200
    for _ in range(reps):
        t = rng.integers(0, 2, 3000)
        y = rng.normal(size=3000)  # null: no effect
        fp += msprt(y, t, tau=0.1)["stopped_at"] is not None
    assert fp / reps < 0.08


def test_msprt_detects_real_effect():
    rng = np.random.default_rng(1)
    t = rng.integers(0, 2, 5000)
    y = rng.normal(size=5000) + 0.3 * t
    assert msprt(y, t, tau=0.3)["stopped_at"] is not None


@pytest.mark.parametrize("cls", [TLearner, XLearner, DRLearner])
def test_learners_recover_heterogeneity(cls):
    df, tau = make_uplift_data(n=12000, seed=2)
    feats = [c for c in df.columns if c.startswith("x")]
    tr, te = df.index < 8000, df.index >= 8000
    m = cls().fit(df.loc[tr, feats], df.t[tr], df.y[tr])
    pred = m.predict(df.loc[te, feats])
    assert np.corrcoef(pred, tau[te])[0, 1] > 0.6
    assert auuc(df.y[te], df.t[te], pred) > 0
    assert auuc(df.y[te], df.t[te], pred) > auuc(df.y[te], df.t[te], np.random.default_rng(0).normal(size=te.sum()))


def test_oracle_ranking_has_monotone_deciles():
    df, tau = make_uplift_data(n=30000, seed=3)
    d = uplift_by_decile(df.y, df.t, tau)
    assert d.obs_uplift.iloc[0] > d.obs_uplift.iloc[-1]
    f, g, r = qini_curve(df.y, df.t, tau)
    assert g[-1] == pytest.approx(r[-1])


def test_policy_gain_matches_ground_truth_in_expectation():
    from causalkit import policy_curve
    df, tau = make_uplift_data(n=200000, seed=5)
    c = policy_curve(df.y, df.t, tau, cost=0.0)   # oracle ranking
    # truth: mean of tau over top-frac share, times frac
    k = int(0.5 * len(tau)); true = np.sort(tau)[::-1][:k].sum() / len(tau)
    est = c.loc[np.isclose(c.frac, 0.5), "gain"].iloc[0]
    assert est == pytest.approx(true, abs=0.03)


def test_policy_honest_when_scores_are_random():
    """Random scores + costly treatment: learned policy must not claim a big win over nothing."""
    from causalkit import evaluate_policy
    rng = np.random.default_rng(0)
    n = 20000
    t = rng.integers(0, 2, n); y = rng.normal(size=n) + 0.1 * t
    r = evaluate_policy(y, t, rng.normal(size=n), cost=0.3, n_boot=100)
    assert r["policy_ci"][1] < 0.1 and r["treat_all_gain"] < 0


def test_policy_beats_treat_all_when_effect_is_heterogeneous():
    from causalkit import evaluate_policy
    df, tau = make_uplift_data(n=30000, seed=6)
    r = evaluate_policy(df.y, df.t, tau, cost=0.5, n_boot=100)
    assert r["policy_minus_all_ci"][0] > 0


def test_calibration_slope_is_one_for_oracle_scores():
    from causalkit import calibration_test
    df, tau = make_uplift_data(n=60000, seed=11)
    r = calibration_test(df.y, df.t, tau)
    assert r["slope_ci"][0] < 1 < r["slope_ci"][1]
    assert r["p_slope_zero"] < 1e-6


def test_calibration_rejects_nothing_for_noise_scores():
    from causalkit import calibration_test
    rng = np.random.default_rng(0); hits, reps = 0, 200
    for s in range(reps):
        n = 3000
        t = rng.integers(0, 2, n); y = rng.normal(size=n) + 0.3 * t   # constant effect
        hits += calibration_test(y, t, rng.normal(size=n))["p_slope_zero"] < 0.05
    assert hits / reps < 0.10


def test_calibration_no_false_heterogeneity_from_fitted_model():
    """Constant true effect: a learner will find 'patterns' in noise, but out-of-fold
    calibration should rarely call the heterogeneity real (nominal 5%)."""
    from sklearn.model_selection import KFold
    from causalkit import calibration_test
    hits, reps = 0, 40
    for s in range(reps):
        rng = np.random.default_rng(s); n = 3000
        X = rng.normal(size=(n, 3)); t = rng.integers(0, 2, n)
        y = X[:, 0] + 0.3 * t + rng.normal(size=n)
        oof = np.zeros(n)
        for tr, te in KFold(2, shuffle=True, random_state=0).split(X):
            oof[te] = TLearner().fit(X[tr], t[tr], y[tr]).predict(X[te])
        hits += calibration_test(y, t, oof)["p_slope_zero"] < 0.05
    assert hits / reps < 0.15


def test_group_effects_gap_and_bootstrap_auuc():
    from causalkit import group_effects, bootstrap_auuc, compare_auuc
    df, tau = make_uplift_data(n=30000, seed=12)
    tbl, gap = group_effects(df.y, df.t, tau)
    assert gap["ci"][0] > 0 and tbl.effect.iloc[0] > tbl.effect.iloc[-1]
    rnd = np.random.default_rng(1).normal(size=len(df))
    assert bootstrap_auuc(df.y, df.t, tau, n_boot=60)["ci"][0] > 0
    lo, hi = bootstrap_auuc(df.y, df.t, rnd, n_boot=60)["ci"]
    assert lo < 0 < hi
    assert compare_auuc(df.y, df.t, tau, rnd, n_boot=60)["distinguishable"]
