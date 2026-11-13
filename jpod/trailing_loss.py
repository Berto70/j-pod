"""Estimate stellar FWHM and asteroid trailing loss."""

import numpy as np
import astropy.units as u
from astropy.table import Column

from .image_analysis import measure_image
from .image_geometry import ImageInfo


def compute_fwhm(image_info: ImageInfo) -> float | None:
    """Return FWHM from the shared image-measurement pass."""
    return measure_image(image_info).fwhm_pixels


def add_trailing_loss(
    table,
    fwhm_pixels: float | None,
    a: float = 0.42,
    b: float = 0.0,
) -> None:
    """Jones+18 use dm = 1.25 log10(1 + a x^2 / (1 + b x)),   x = trail length / FWHM
    (a, b) = (0.67, 1.16): SNR loss;  (0.42, 0.0): detection loss with a PSF-like filter
    
    Add trail/FWHM, trailing loss, and trail-adjusted V magnitude."""
    trail_pixels = np.asarray(table["trail_length [px]"], dtype=float)
    if fwhm_pixels is None:
        ratio = np.full(len(table), np.nan)
        delta_V = np.full(len(table), np.nan)
    else:
        if not np.isfinite(fwhm_pixels) or fwhm_pixels <= 0:
            raise ValueError("FWHM must be a positive finite value in pixels.")

        ratio = trail_pixels / fwhm_pixels
        delta_V = 1.25 * np.log10(1 + (a * ratio**2) / (1 + b * ratio))

    assert np.all(delta_V[np.isfinite(delta_V)] >= 0)

    visual_magnitude = np.asarray(
        u.Quantity(table["V"]).to_value(u.mag),
        dtype=float,
    )
    trail_adjusted_magnitude = visual_magnitude + delta_V
    for name, values, unit in (
        ("trail_length / FWHM", ratio, None),
        ("trailing_loss", delta_V, u.mag),
        ("V_trailing", trail_adjusted_magnitude, u.mag),
    ):
        column = Column(values, name=name, unit=unit)
        if name in table.colnames:
            table.replace_column(name, column)
        else:
            table.add_column(column)
