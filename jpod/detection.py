"""Detection probability: P_det = P_pos * P_mag.

P_pos : probability that the true position (2D Gaussian built from the JPL
        uncertainty ellipse) falls inside the image.
P_mag : probability that the true magnitude (Gaussian around V, width sigma_V)
        is brighter than the limiting magnitude V_lim.
"""

import astropy.units as u
import numpy as np
from scipy.stats import multivariate_normal, norm

MIN_VARIANCE = 1e-6   # pixel^2, avoids singular covariances for tiny ellipses


def pixel_covariance(sigma_a, sigma_b, angle_deg):
    """Covariance matrix (pixel^2) of a Gaussian whose 1-sigma ellipse has
    semi-axes sigma_a, sigma_b (pixels) and major axis at angle_deg
    (CCW from +x)."""
    th = np.radians(angle_deg)
    c, s = np.cos(th), np.sin(th)
    rot = np.array([[c, -s], [s, c]])
    return rot @ np.diag([sigma_a**2, sigma_b**2]) @ rot.T


def position_probability(mu, cov, nx, ny, eps=1e-8):
    """Probability that a 2D Gaussian (mean mu, covariance cov, in pixels)
    falls inside the image rectangle.
    """
    xmin, xmax, ymin, ymax = -0.5, nx - 0.5, -0.5, ny - 0.5
    cov = np.asarray(cov, dtype=float) + MIN_VARIANCE * np.eye(2)

    def cdf(x, y):
        return multivariate_normal.cdf([x, y], mean=mu, cov=cov)

    p = cdf(xmax, ymax) - cdf(xmin, ymax) - cdf(xmax, ymin) + cdf(xmin, ymin)
    return p


def magnitude_probability(v, v_lim, sigma_v=0.5):
    """P(V_true <= V_lim) for V_true ~ N(v, sigma_v^2). NaN if v is NaN.

    All arguments are reduced to plain floats in magnitudes before the
    subtraction, so passing a Quantity (``21.5 * u.mag``), a dimensionless
    or unit-ed table column never raises a UnitConversionError.
    """
    v = _mag_array(v)
    v_lim = _mag_float(v_lim)
    sigma_v = _mag_float(sigma_v)
    with np.errstate(invalid="ignore"):
        return norm.cdf((v_lim - v) / sigma_v)


def _plain(column):
    """Column -> float ndarray (units stripped), masked values -> NaN."""
    values = column
    unit = getattr(values, "unit", None)
    if unit is not None:                       # Quantity / SkyCoord-like
        values = values.to_value(unit)         # plain numbers in that unit
    arr = np.ma.asarray(values, dtype=float)   # no copy, keeps the mask
    return np.ma.filled(arr, np.nan).view(np.ndarray)


def _mag_array(value):
    """Any magnitude input -> plain float ndarray (NaN for missing)."""
    return np.atleast_1d(_plain(value)).astype(float, copy=False)


def _mag_float(value):
    """Scalar magnitude (float or Quantity in mag) -> plain float."""
    if isinstance(value, u.Quantity):
        return value.to_value(u.mag)
    return float(value)


def add_detection_probability(table, info, v_lim, sigma_v=0.5,
                              ellipse_nsigma=3.0, posunc_nsigma=1.0):
    """Add the columns P_pos, P_mag and P_det.

    The table needs x_pix, y_pix, SMAA_3sigma, SMIA_3sigma, ellipse_angle,
    posunc and V. The JPL ellipse is a 3-sigma ellipse (ellipse_nsigma).
    Objects without ellipse use an isotropic Gaussian with sigma =
    posunc / posunc_nsigma (check how SkyBoT defines posunc and set
    posunc_nsigma accordingly). Objects with unknown V get NaN in P_mag and
    P_det.
    """
    scale = info.pixel_scale
    n = len(table)

    x = _plain(table["x_pix"])
    y = _plain(table["y_pix"])
    a = u.Quantity(table["SMAA_3sigma"]).to_value(u.arcsec) / ellipse_nsigma / scale
    b = u.Quantity(table["SMIA_3sigma"]).to_value(u.arcsec) / ellipse_nsigma / scale
    ang = u.Quantity(table["ellipse_angle"]).to_value(u.deg)
    p = u.Quantity(table["posunc"]).to_value(u.arcsec) / posunc_nsigma / scale

    p_pos = np.empty(n)
    for i in range(n):
        if np.all(np.isfinite([a[i], b[i], ang[i]])):
            cov = pixel_covariance(a[i], b[i], ang[i])
        else:
            r = p[i] if np.isfinite(p[i]) else 0.0
            cov = r**2 * np.eye(2)
        p_pos[i] = position_probability((x[i], y[i]), cov, info.nx, info.ny)

    p_mag = magnitude_probability(table["V"], v_lim, sigma_v)

    table["P_pos"] = p_pos
    table["P_mag"] = p_mag
    table["P_det"] = p_pos * p_mag
    for name in ("P_pos", "P_mag", "P_det"):
        table[name].info.format = ".3g"
    return table
