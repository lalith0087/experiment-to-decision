"""Experiment health checks."""
import numpy as np
from scipy.stats import chisquare


def srm_test(counts, expected_ratios=None, threshold=0.001):
    """Sample Ratio Mismatch chi-square test.

    Returns dict with p-value and `srm` flag (True => assignment looks broken).
    Conventional threshold is strict (0.001) because SRM invalidates the test.
    """
    counts = np.asarray(counts, dtype=float)
    k = len(counts)
    ratios = np.full(k, 1 / k) if expected_ratios is None else np.asarray(expected_ratios, float)
    ratios = ratios / ratios.sum()
    stat, p = chisquare(counts, ratios * counts.sum())
    return {"chi2": float(stat), "p_value": float(p), "srm": bool(p < threshold)}
