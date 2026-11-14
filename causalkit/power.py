"""Power analysis for two-arm experiments (two-sided z-test)."""
import numpy as np
from scipy.stats import norm


def _z(alpha, power):
    return norm.ppf(1 - alpha / 2), norm.ppf(power)


def sample_size_mean(mde, sd, alpha=0.05, power=0.8, ratio=1.0):
    """Per-control-arm n to detect absolute difference `mde` in a mean.

    `ratio` = n_treatment / n_control. Returns (n_control, n_treatment).
    """
    za, zb = _z(alpha, power)
    n_c = (sd**2) * (1 + 1 / ratio) * (za + zb) ** 2 / mde**2
    n_c = int(np.ceil(n_c))
    return n_c, int(np.ceil(n_c * ratio))


def sample_size_proportion(p_base, mde, alpha=0.05, power=0.8, ratio=1.0):
    """Sample sizes to detect absolute lift `mde` on a baseline rate."""
    za, zb = _z(alpha, power)
    p1, p2 = p_base, p_base + mde
    var = p1 * (1 - p1) + p2 * (1 - p2) / ratio
    n_c = int(np.ceil(var * (za + zb) ** 2 / mde**2))
    return n_c, int(np.ceil(n_c * ratio))


def mde_mean(n_control, sd, alpha=0.05, power=0.8, ratio=1.0):
    """Minimum detectable absolute effect for a given control-arm size."""
    za, zb = _z(alpha, power)
    return (za + zb) * sd * np.sqrt((1 + 1 / ratio) / n_control)
