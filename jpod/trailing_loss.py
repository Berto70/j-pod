"""Estimate stellar FWHM and asteroid trailing loss."""

import numpy as np
import astropy.units as u
from astropy.table import Column
from scipy.ndimage import maximum_filter
from scipy.optimize import curve_fit
from photutils.background import Background2D, MedianBackground

from .image_geometry import ImageInfo


def compute_fwhm(image_info: ImageInfo) -> float:
    """Return stellar FWHM in pixels, using the FITS value when available."""
    header_fwhm = image_info.header_fwhm
    if header_fwhm is not None:
        if not np.isfinite(header_fwhm) or header_fwhm <= 0:
            raise ValueError("FITS header FWHM must be a positive finite value in pixels.")
        return float(header_fwhm)

    data = np.asarray(image_info.data, dtype=float)

    nan_mask = np.isnan(data)

    bkg = Background2D(
        data,
        box_size = (50, 50),
        filter_size=(3, 3),
        mask = nan_mask, # exclude mask from background estimation
        bkg_estimator=MedianBackground()
    )
    background = bkg.background

    image = data - background
    noise = 1.4826 * np.median(np.abs(image - np.median(image)))
    if not np.isfinite(noise) or noise <= 0:
        raise RuntimeError("Could not estimate image noise for stellar detection.")

    # 1. Find local maxima whose background-subtracted peak exceeds 5 sigma.
    peak_mask = (image == maximum_filter(image, size=9)) & (image > 5 * noise)
    peak_positions = np.argwhere(peak_mask)

    def gaussian(coords, amplitude, x0, y0, sigma_major, sigma_minor, theta, offset):
        x, y = coords
        dx = x - x0
        dy = y - y0
        major = dx * np.cos(theta) + dy * np.sin(theta)
        minor = -dx * np.sin(theta) + dy * np.cos(theta)
        return offset + amplitude * np.exp(
            -0.5 * ((major / sigma_major) ** 2 + (minor / sigma_minor) ** 2)
        )

    fitted_sources = []
    image_height, image_width = image.shape
    time_start = time.time()
    for y, x in peak_positions:
        # A 11x11 cutout is needed to fit the full PSF, including its wings.
        if x < 5 or y < 5 or x >= image_width - 5 or y >= image_height - 5:
            continue
        cutout = image[y - 5:y + 6, x - 5:x + 6]
        yy, xx = np.indices(cutout.shape)
        try:
            parameters, _ = curve_fit(
                gaussian,
                (xx.ravel(), yy.ravel()),
                cutout.ravel(),
                p0=(cutout.max(), 5, 5, 1.5, 1.5, 0, 0),
                bounds=(
                    [0, 3, 3, 0.4, 0.4, -np.pi / 2, -np.inf],
                    [np.inf, 7, 7, 4.5, 4.5, np.pi / 2, np.inf],
                ),
                maxfev=3000,
            )
        except (RuntimeError, ValueError):
            continue

        amplitude, _, _, sigma_major, sigma_minor, theta, _ = parameters
        if not np.all(np.isfinite(parameters)) or amplitude <= 0:
            continue
        fwhm = 2.35482 * np.sqrt(sigma_major * sigma_minor)
        elongation = max(sigma_major, sigma_minor) / min(sigma_major, sigma_minor)

        # 2. Keep high-SNR, compact, round profiles; reject obvious elongation.
        dx = xx - parameters[1]
        dy = yy - parameters[2]
        major = dx * np.cos(theta) + dy * np.sin(theta)
        minor = -dx * np.sin(theta) + dy * np.cos(theta)
        profile = np.exp(
            -0.5 * ((major / sigma_major) ** 2 + (minor / sigma_minor) ** 2)
        )
        snr = amplitude * np.sqrt(np.sum(profile**2)) / noise
        if snr < 20 or elongation > 1.3:
            continue
        fitted_sources.append((x, y, fwhm))
    time_end = time.time()
    print(time_end - time_start, "seconds to fit")
    if not fitted_sources:
        raise RuntimeError("No unsaturated, high-SNR stellar profiles were found.")

    # 3. Reject blends: nearby 5-sigma peaks can distort each other's fitted PSF.
    positions = np.asarray([(source[0], source[1]) for source in fitted_sources])
    widths = np.asarray([source[2] for source in fitted_sources])
    separations = np.hypot(
        positions[:, np.newaxis, 0] - positions[np.newaxis, :, 0],
        positions[:, np.newaxis, 1] - positions[np.newaxis, :, 1],
    )
    blend_limit = 2 * np.maximum(widths[:, np.newaxis], widths[np.newaxis, :])
    blended = np.any((separations < blend_limit) & (separations > 0), axis=1)
    stellar_fwhm = widths[~blended]
    if stellar_fwhm.size == 0:
        raise RuntimeError("No isolated stellar profiles remain after blend rejection.")

    # 4. Iteratively reject extended outliers, then take the median stellar FWHM.
    for _ in range(3):
        median_fwhm = np.median(stellar_fwhm)
        scatter = 1.4826 * np.median(np.abs(stellar_fwhm - median_fwhm))
        if scatter == 0:
            break
        selected = np.abs(stellar_fwhm - median_fwhm) <= 3 * scatter
        if np.all(selected):
            break
        stellar_fwhm = stellar_fwhm[selected]
    if stellar_fwhm.size == 0:
        raise RuntimeError("No stellar FWHM measurements remain after outlier rejection.")
    return float(np.median(stellar_fwhm))


def add_trailing_loss(table, fwhm_pixels: float) -> None:
    """Add trail/FWHM, trailing loss, and trail-adjusted V magnitude."""
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
