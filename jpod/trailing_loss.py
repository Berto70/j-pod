"""Estimate stellar FWHM and asteroid trailing loss."""

import numpy as np
import astropy.units as u
from astropy.table import Column
from scipy.ndimage import maximum_filter
from scipy.optimize import curve_fit
from scipy.spatial.distance import cdist
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
    if np.any(nan_mask):
        fill_val = np.nanmedian(data)
        data[nan_mask] = fill_val

    bkg = Background2D(
        data,
        box_size = (50, 50),
        filter_size=(3, 3),
        # mask = nan_mask, # exclude mask from background estimation
        bkg_estimator=MedianBackground()
    )
    background = bkg.background

    image = data - background
    noise = 1.4826 * np.median(np.abs(image - np.median(image)))
    if not np.isfinite(noise) or noise <= 0:
        raise RuntimeError("Could not estimate image noise for stellar detection.")

    # image[np.isnan(image)] = 0.0

    # 1. Find local maxima whose background-subtracted peak exceeds 5 sigma.
    peak_mask = (image == maximum_filter(image, size=9)) & (image > 5 * noise)
    peak_positions = np.argwhere(peak_mask)

    # we only need the median FWHM (see Vernes+2012 sec 2.2)
    # --> we fit only 100 brightest stars
    peak_values = image[peak_mask]
    sorted_idx = np.argsort(peak_values)[::-1]
    peak_positions = peak_positions[sorted_idx][:100]

    fitted_sources = []
    image_height, image_width = image.shape

    yy, xx = np.indices((11,11))

    for y, x in peak_positions:
        # A 11x11 cutout is needed, including its wings.
        if x < 5 or y < 5 or x >= image_width - 5 or y >= image_height - 5:
            continue
        cutout = image[y - 5:y + 6, x - 5:x + 6]

        yy, xx = np.indices(cutout.shape)
        
        # 2nd.order moments provide exact solution for Gaussian widths (see Vernes+2012 sec 2.2)
        total_flux = np.sum(cutout)
        if total_flux <= 0:
            continue

        x_c = np.sum(xx * cutout) / total_flux
        y_c = np.sum(yy * cutout) / total_flux

        dx = xx - x_c
        dy = yy - y_c
        # 2nd order moments
        mu20 = np.sum(dx**2 * cutout) / total_flux
        mu02 = np.sum(dy**2 * cutout) / total_flux
        mu11 = np.sum(dx * dy * cutout) / total_flux

        # eigenvalues of the covariance matrix
        diff = mu20 - mu02
        disc = np.sqrt(diff**2 + 4 * mu11**2)
        lambda1 = 0.5 * (mu20 + mu02 + disc) # semi-major axis variance
        lambda2 = 0.5 * (mu20 + mu02 - disc) # semi-minor axis variance

        lambda1 = max(0.0, lambda1) # prevents NaNs in sqrt
        lambda2 = max(0.0, lambda2)

        # reject 1D profiles or invalid shapes
        if lambda1 <= 0 or lambda2 <=0:
            continue

        sigma_major = np.sqrt(lambda1)
        sigma_minor = np.sqrt(lambda2)

        # geometric mean 
        fwhm = 2.35482 * np.sqrt(sigma_major * sigma_minor)
        elongation = max(sigma_major, sigma_minor) / min(sigma_major, sigma_minor)

        # for a 2D-Gaussian, optimal SNR --> tot_flux / (2*noise*sqrt(pi*sigma_x*sigma_y))
        snr = total_flux / (2 * noise * np.sqrt(np.pi * sigma_major * sigma_minor))
        if snr < 20 or elongation > 1.3:
            continue
        fitted_sources.append((x, y, fwhm))
        
    if not fitted_sources:
        raise RuntimeError("No unsaturated, high-SNR stellar profiles were found.")

    # 3. Reject blends: nearby 5-sigma peaks can distort each other's fitted PSF.
    positions = np.asarray([(source[0], source[1]) for source in fitted_sources])
    widths = np.asarray([source[2] for source in fitted_sources])

    separations = cdist(positions, positions, metric="euclidean")

    blend_limit = 2 * np.maximum(widths[:, np.newaxis], widths[np.newaxis, :])
    blended = np.any((separations < blend_limit) & (separations > 0), axis=1)
    stellar_fwhm = widths[~blended]
    
    if stellar_fwhm.size == 0:
        raise RuntimeError("No isolated stellar profiles remain after blend rejection.")

    # 4. Iteratively reject extended outliers, then take the median stellar FWHM.
    for _ in range(3):

        if stellar_fwhm.size == 0:
            break

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
