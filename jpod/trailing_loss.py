"""Estimate stellar FWHM and asteroid trailing loss."""

import numpy as np
import astropy.units as u
from astropy.table import Column

from .image_analysis import measure_image
from .image_geometry import ImageInfo


def compute_fwhm(image_info: ImageInfo) -> float | None:
    """Return FWHM from the shared image-measurement pass."""
    return measure_image(image_info).fwhm_pixels


def add_trailing_loss(table, fwhm_pixels: float | None) -> None:
    """Add trail corrections, or unavailable values when FWHM is unknown."""
    if fwhm_pixels is None:
        trail_pixels = np.full(len(table), np.nan)
        ratio = np.full(len(table), np.nan)
        delta_m = np.full(len(table), np.nan)
        trail_adjusted_magnitude = np.full(len(table), np.nan)
    else:
        if not np.isfinite(fwhm_pixels) or fwhm_pixels <= 0:
            raise ValueError("FWHM must be a positive finite value in pixels.")

        trail_pixels = np.asarray(table["trail_length [px]"], dtype=float)
        ratio = trail_pixels / fwhm_pixels
        delta_m = 2.5 * np.log10(1 + ratio)
        visual_magnitude = np.asarray(
            u.Quantity(table["V"]).to_value(u.mag),
            dtype=float,
        )
        trail_adjusted_magnitude = visual_magnitude + delta_m

    for name, values, unit in (
        ("trail_length / FWHM", ratio, None),
        ("trailing_loss", delta_m, u.mag),
        ("V_trailing", trail_adjusted_magnitude, u.mag),
    ):
        column = Column(values, name=name, unit=unit)
        if name in table.colnames:
            table.replace_column(name, column)
        else:
            table.add_column(column)
