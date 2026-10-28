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

def bounds_inside_both_ends(nx, ny, dx, dy):
    """Rectangle (xmin, xmax, ymin, ymax) of the start position for which the object is inside the image
    both at the start and at the end of the exposure.
    The object moves by (dx,dy) pixels during the exposure. It is inside the image at the start if xmin <= x <= xmax (same for y),
    and at the end if xmin <= x+dx <= xmax. Both conditions together give x in [max(xmin, xmin-dx), min(xmax, xmax-dx)]: the image rectangle 
    intersected with itself shifted back by the motion, which is again a rectangle. 
    """
    xmin, xmax, ymin, ymax = -0.5, nx-0.5, -0.5, ny - 0.5
    return (max(xmin,xmin-dx), min(xmax, xmax-dx),
            max(ymin, ymin-dy), min(ymax, ymax-dy))

def position_probability(mu, cov, xmin, xmax, ymin, ymax):
    """Probability that a 2D Gaussian (mean mu, covariance cov, in pixels)
    falls inside the rectangle xmin <= x <= xmax, ymin <= y <= ymax. 
    Uses inclusion-exclusion on the bivariate CDF
    """
    if xmax <= xmin or ymax <= ymin:
        return 0.0
    cov = np.asarray(cov, dtype=float) + MIN_VARIANCE*np.eye(2)
   
    def cdf(x, y):
        return multivariate_normal.cdf([x, y], mean=mu, cov=cov)

    p = cdf(xmax, ymax) - cdf(xmin, ymax) - cdf(xmax, ymin) + cdf(xmin, ymin)
    return p


def magnitude_probability(v, v_lim, sigma_v=0.5):
    """P(V_true <= V_lim) for V_true ~ N(v, sigma_v^2). NaN if v is NaN."""
    return norm.cdf((v_lim - v) / sigma_v)


def _plain(column):
    """Column -> float array, with masked values replaced by NaN."""
    return np.ma.filled(np.ma.asarray(column, dtype=float), np.nan)


def add_detection_probability(table, info, v_lim, sigma_v=0.5,
                              ellipse_nsigma=3.0, posunc_nsigma=1.0):
    """Add the columns P_pos, P_mag and P_det.

    The table needs x_pix, y_pix (position at the start of the exposure), SMAA_3sigma, SMIA_3sigma, ellipse_angle,
    posunc and V. The JPL ellipse is a 3-sigma ellipse (ellipse_nsigma). If it also has x_pix_end, y_pix_end 
    (position at the end of the exposure), P_pos is the probability of being inside the image at both times; otherwise
    it is the probability of being inside at the single position given.
    
    The JPL ellipse is a 3-sigma ellipse. Objects without ellipse use an isotropic Gaussian with sigma =
    posunc / posunc_nsigma.
    Objects with unknown V get NaN in P_mag and P_det.
    """
    scale = info.pixel_scale
    n = len(table)

    x = _plain(table["x_pix"])
    y = _plain(table["y_pix"])
    if "x_pix_end" in table.colnames and "y_pix_end" in table.colnames:
        x_end = _plain(table["x_pix_end"])
        y_end = _plain(table["y_pix_end"])
    else:
        x_end, y_end = x, y
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
        # Gaussian centered on the start position; the area to integrate is the set of start positions that keep the object inside
        # the image until the end of the exposure
        bounds = bounds_inside_both_ends(info.nx, info.ny,
                                         x_end[i]-x[i], y_end[i]-y[i])
        p_pos[i] = position_probability((x[i], y[i]), cov, *bounds)

    p_mag = magnitude_probability(_plain(table["V"]), v_lim, sigma_v)

    table["P_pos"] = p_pos
    table["P_mag"] = p_mag
    table["P_det"] = p_pos * p_mag
    for name in ("P_pos", "P_mag", "P_det"):
        table[name].info.format = ".3g"
    return table
