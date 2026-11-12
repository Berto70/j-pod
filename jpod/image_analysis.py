"""Shared background, stellar-profile, FWHM, and aperture measurements."""

from dataclasses import dataclass

from astropy.coordinates import SkyCoord
import numpy as np
from scipy.ndimage import maximum_filter
from scipy.spatial.distance import cdist
from photutils.background import Background2D, MedianBackground

from .image_geometry import ImageInfo

_BACKGROUND_BOX_SIZE = 50
_PEAK_FILTER_SIZE = 9
_PROFILE_HALF_SIZE = 5
_MAX_PROFILE_CANDIDATES = 200
_MAX_CALIBRATION_STARS = 50
_MIN_PROFILE_SNR = 20.0
_MAX_ELONGATION = 1.3


@dataclass(frozen=True)
class ImageStar:
    """A vetted stellar profile and its background-subtracted aperture flux."""

    x: float
    y: float
    flux: float
    peak: float
    fwhm_pixels: float
    ra_deg: float
    dec_deg: float


@dataclass(frozen=True)
class ImageMeasurements:
    """Measurements shared by FWHM reporting and photometric calibration."""

    fwhm_pixels: float | None
    background: np.ndarray
    noise: float
    stars: tuple[ImageStar, ...]


def measure_image(image_info: ImageInfo) -> ImageMeasurements:
    """Estimate the background once and reuse vetted stars for FWHM and ZP.

    Star fluxes use a circular aperture with radius equal to the selected
    FWHM. A header FWHM is preferred when present; otherwise the median of
    isolated, round, high-SNR stellar profiles supplies the FWHM.
    """
    data = np.asarray(image_info.data, dtype=float)
    if data.ndim != 2:
        raise ValueError("FWHM and zero-point measurements require a 2-D image.")

    valid = np.isfinite(data)
    if not np.any(valid):
        raise RuntimeError("Cannot measure an image containing no finite pixels.")

    # Background2D needs finite input; masked pixels are filled only for its
    # numerical grid and are excluded from all measurements below.
    safe_data = np.where(valid, data, np.nanmedian(data[valid]))
    box_size = tuple(min(_BACKGROUND_BOX_SIZE, size) for size in data.shape)
    background = Background2D(
        safe_data,
        box_size=box_size,
        filter_size=(3, 3),
        mask=~valid,
        bkg_estimator=MedianBackground(),
        exclude_percentile=100,
    ).background

    # Remove the spatial background before finding peaks and integrating flux.
    residual = safe_data - background
    residual[~valid] = 0.0
    valid_residual = residual[valid]
    noise = float(
        1.4826 * np.median(np.abs(valid_residual - np.median(valid_residual)))
    )
    if not np.isfinite(noise) or noise <= 0:
        return _measurements_without_stars(image_info, background, noise)

    # Keep the same peak threshold and candidate count used by the prior FWHM
    # estimator, so its established FWHM behavior remains familiar.
    peak_mask = (
        (residual == maximum_filter(residual, size=_PEAK_FILTER_SIZE))
        & (residual > 5.0 * noise)
    )
    peak_y, peak_x = np.nonzero(peak_mask)
    if peak_x.size == 0:
        return _measurements_without_stars(image_info, background, noise)

    candidate_order = np.argsort(residual[peak_y, peak_x])[::-1][
        :_MAX_PROFILE_CANDIDATES
    ]
    image_height, image_width = data.shape
    profile_stars = []

    # Moments estimate each compact profile's width and centroid from the same
    # background-subtracted pixels later used by aperture photometry.
    for candidate in candidate_order:
        x_peak = int(peak_x[candidate])
        y_peak = int(peak_y[candidate])
        half_size = _PROFILE_HALF_SIZE
        if (
            x_peak < half_size
            or y_peak < half_size
            or x_peak >= image_width - half_size
            or y_peak >= image_height - half_size
        ):
            continue

        # Saturated cores distort both moments and flux calibration.
        if (
            image_info.header_saturation is not None
            and data[y_peak, x_peak] >= image_info.header_saturation
        ):
            continue

        y0, y1 = y_peak - half_size, y_peak + half_size + 1
        x0, x1 = x_peak - half_size, x_peak + half_size + 1
        cutout = residual[y0:y1, x0:x1]
        cutout_valid = valid[y0:y1, x0:x1]
        yy, xx = np.indices(cutout.shape, dtype=float)
        # Negative noise is not useful as a centroid weight, but remains in
        # the aperture sum to avoid a positive-flux bias.
        weights = np.where(cutout_valid, np.maximum(cutout, 0.0), 0.0)
        total_weight = float(weights.sum())
        if total_weight <= 0:
            continue

        x_local = float((weights * xx).sum() / total_weight)
        y_local = float((weights * yy).sum() / total_weight)
        dx = xx - x_local
        dy = yy - y_local
        moment_xx = float((weights * dx**2).sum() / total_weight)
        moment_yy = float((weights * dy**2).sum() / total_weight)
        moment_xy = float((weights * dx * dy).sum() / total_weight)
        covariance = np.array([[moment_xx, moment_xy], [moment_xy, moment_yy]])
        eigenvalues = np.linalg.eigvalsh(covariance)
        if eigenvalues[0] <= 0 or eigenvalues[1] <= 0:
            continue

        sigma_minor, sigma_major = np.sqrt(eigenvalues)
        elongation = float(sigma_major / sigma_minor)
        if elongation > _MAX_ELONGATION:
            continue

        # The geometric-mean width makes the FWHM insensitive to PSF rotation.
        profile_fwhm = float(2.35482 * np.sqrt(sigma_major * sigma_minor))
        peak = float(residual[y_peak, x_peak])
        profile_area = 2.0 * np.pi * sigma_major * sigma_minor
        profile_snr = total_weight / (noise * np.sqrt(profile_area))
        if profile_snr < _MIN_PROFILE_SNR:
            continue

        # Without a saturation keyword, reject an obvious flat-topped core.
        if image_info.header_saturation is None:
            core = data[y_peak - 1:y_peak + 2, x_peak - 1:x_peak + 2]
            if np.count_nonzero(core == data[y_peak, x_peak]) > 1:
                continue

        profile_stars.append(
            (x0 + x_local, y0 + y_local, peak, profile_fwhm)
        )

    # Nearby profiles can bias FWHM and catalog fluxes through blending.
    profile_stars = _reject_blended_profiles(profile_stars)
    if not profile_stars:
        return _measurements_without_stars(image_info, background, noise)

    measured_fwhm = _robust_median_fwhm(
        np.asarray([star[3] for star in profile_stars], dtype=float)
    )
    fwhm_pixels = _select_fwhm(image_info.header_fwhm, measured_fwhm)

    # Use the selected FWHM consistently for every calibration-star aperture.
    aperture_radius_squared = fwhm_pixels**2
    calibrated_stars = []
    for x_center, y_center, peak, profile_fwhm in profile_stars:
        half_window = max(_PROFILE_HALF_SIZE, int(np.ceil(1.5 * fwhm_pixels)))
        x_peak = int(round(x_center))
        y_peak = int(round(y_center))
        if (
            x_peak < half_window
            or y_peak < half_window
            or x_peak >= image_width - half_window
            or y_peak >= image_height - half_window
        ):
            continue

        y0, y1 = y_peak - half_window, y_peak + half_window + 1
        x0, x1 = x_peak - half_window, x_peak + half_window + 1
        cutout = residual[y0:y1, x0:x1]
        cutout_valid = valid[y0:y1, x0:x1]
        yy, xx = np.ogrid[y0:y1, x0:x1]
        aperture = (
            (xx - x_center) ** 2 + (yy - y_center) ** 2
            <= aperture_radius_squared
        )
        flux = float(np.sum(cutout[aperture & cutout_valid]))
        if not np.isfinite(flux) or flux <= 0:
            continue

        sky_coord = SkyCoord.from_pixel(x_center, y_center, image_info.wcs)
        calibrated_stars.append(
            ImageStar(
                float(x_center),
                float(y_center),
                flux,
                peak,
                profile_fwhm,
                float(sky_coord.ra.deg),
                float(sky_coord.dec.deg),
            )
        )

    # FWHM used every vetted profile above; keep the brightest 50 aperture
    # measurements for the catalog match to bound matching and memory costs.
    calibrated_stars.sort(key=lambda star: star.flux, reverse=True)
    calibrated_stars = calibrated_stars[:_MAX_CALIBRATION_STARS]
    return ImageMeasurements(
        fwhm_pixels,
        background,
        noise,
        tuple(calibrated_stars),
    )


def _measurements_without_stars(
    image_info: ImageInfo,
    background: np.ndarray,
    noise: float,
) -> ImageMeasurements:
    """Keep background measurements even when no usable FWHM is available."""
    fwhm_pixels = image_info.header_fwhm
    if fwhm_pixels is not None:
        fwhm_pixels = float(fwhm_pixels)
        if not np.isfinite(fwhm_pixels) or fwhm_pixels <= 0:
            fwhm_pixels = None
    return ImageMeasurements(fwhm_pixels, background, noise, ())


def _select_fwhm(header_fwhm: float | None, measured_fwhm: float) -> float:
    """Prefer a plausible header FWHM but reject values off by more than 2x."""
    if header_fwhm is None:
        return measured_fwhm

    header_value = float(header_fwhm)
    if not np.isfinite(header_value) or header_value <= 0:
        raise ValueError("FITS header FWHM must be a positive finite value in pixels.")

    ratio = header_value / measured_fwhm
    if ratio > 2.0 or ratio < 0.5:
        print(
            f"FWHM note: the FITS header says {header_value:.2f} px, but "
            f"measuring stars in the image gives {measured_fwhm:.2f} px. "
            "Using the image-measured value."
        )
        return measured_fwhm
    return header_value


def _reject_blended_profiles(
    stars: list[tuple[float, float, float, float]],
) -> list[tuple[float, float, float, float]]:
    """Remove profiles with another detected peak within two local FWHM."""
    if not stars:
        return []

    positions = np.asarray([(star[0], star[1]) for star in stars])
    widths = np.asarray([star[3] for star in stars])
    separations = cdist(positions, positions, metric="euclidean")
    blend_limit = 2.0 * np.maximum(widths[:, np.newaxis], widths[np.newaxis, :])
    blended = np.any(
        (separations < blend_limit) & (separations > 0),
        axis=1,
    )
    return [star for star, is_blended in zip(stars, blended) if not is_blended]


def _robust_median_fwhm(widths: np.ndarray) -> float:
    """Return a median width after iterative 3-MAD rejection."""
    selected_widths = widths
    for _ in range(3):
        if selected_widths.size == 0:
            break
        median = float(np.median(selected_widths))
        scatter = float(
            1.4826 * np.median(np.abs(selected_widths - median))
        )
        if scatter == 0:
            break
        selected = np.abs(selected_widths - median) <= 3.0 * scatter
        if np.all(selected):
            break
        selected_widths = selected_widths[selected]

    if selected_widths.size == 0:
        raise RuntimeError("No stellar FWHM measurements remain after outlier rejection.")
    return float(np.median(selected_widths))
