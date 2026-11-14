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
