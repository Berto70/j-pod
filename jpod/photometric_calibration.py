"""Determine a count-rate photometric zero point from FITS images and V50."""

from dataclasses import dataclass
from pathlib import Path
import warnings

import astropy.units as u
from astropy.coordinates import SkyCoord
import numpy as np
from scipy.spatial import cKDTree

from .image_analysis import ImageMeasurements, ImageStar
from .image_geometry import ImageInfo

_CATALOG_HEADER_SIZE = 110
_V50_RECORD_SIZE = 6
_DEG_PER_RING = 90.0 / 17.5
_MIN_CALIBRATION_STARS = 5


@dataclass(frozen=True)
class ZeroPointResult:
    """A zero point, or the reason it could not be calculated."""

    value: float | None
    source: str
    matched_stars: int = 0
    scatter: float | None = None
    unavailable_reason: str | None = None


@dataclass(frozen=True)
class LimitingMagnitudeResult:
    """A sky-noise limited magnitude and the measurements behind it."""

    value: float | None
    background_per_pixel: float | None = None
    sky_noise: float | None = None
    aperture_area_pixels: float | None = None
    gain_e_per_adu: float | None = None
    unavailable_reason: str | None = None


def determine_zero_point(
    image_info: ImageInfo,
    measurements: ImageMeasurements | None,
    catalog_dir: Path,
) -> ZeroPointResult:
    """Use a header zero point, or derive one by matching image stars to V50.

    The returned zero point uses the convention ``m = ZP - 2.5 log10(rate)``,
    where ``rate`` is the background-subtracted aperture flux per second.
    An unavailable result includes the reason so the caller can report it.
    """
    header_points = dict(image_info.header_zero_points)
    if "MAGZERO" in header_points:
        if (
            "PHOTZP" in header_points
            and not np.isclose(header_points["MAGZERO"], header_points["PHOTZP"])
        ):
            warnings.warn(
                "MAGZERO and PHOTZP differ; using MAGZERO as requested.",
                RuntimeWarning,
                stacklevel=2,
            )
        return ZeroPointResult(header_points["MAGZERO"], "MAGZERO header")
    if "PHOTZP" in header_points:
        return ZeroPointResult(header_points["PHOTZP"], "PHOTZP header")

    # V50 contains Johnson V magnitudes; do not silently use them as another
    # filter's calibration or guess that an unspecified filter is V.
    if not is_v_filter(image_info.filter_name):
        filter_description = (
            f"the image filter is {image_info.filter_name!r}"
            if image_info.filter_name
            else "the image filter is unspecified"
        )
        return _unavailable(
            f"the local ASTAP V50 catalog contains Johnson V magnitudes, but "
            f"{filter_description}. Catalog calibration is currently supported "
            "only for images identified as V band"
        )

    if image_info.exptime is None:
        return _unavailable("exposure time is missing; cannot compute a count-rate zero point")
    exposure_seconds = image_info.exptime.to_value(u.s)
    if not np.isfinite(exposure_seconds) or exposure_seconds <= 0:
        return _unavailable("exposure time must be positive to compute a zero point")
    if not image_info.wcs.has_celestial:
        return _unavailable("celestial WCS is missing; image stars cannot be matched to V50")
    if measurements is None:
        return _unavailable("shared image measurements are unavailable")

    catalog_dir = Path(catalog_dir)
    if not catalog_dir.is_dir():
        return _unavailable(f"V50 catalog directory does not exist: {catalog_dir}")

    # Reuse the detected, vetted, aperture-measured stars from the FWHM pass.
    stars = list(measurements.stars)
    if len(stars) < _MIN_CALIBRATION_STARS:
        return _unavailable(
            f"only {len(stars)} suitable image stars were detected; "
            f"at least {_MIN_CALIBRATION_STARS} are required"
        )

    if measurements.fwhm_pixels is None:
        return _unavailable("FWHM is unavailable and no image stars were measured")
    match_radius_pixels = max(2.0, measurements.fwhm_pixels)
    catalog_files = _catalog_files_for_stars(
        stars,
        image_info.pixel_scale,
        match_radius_pixels,
        catalog_dir,
    )
    existing_files = [path for path in catalog_files if path.is_file()]
    missing_files = sorted(set(catalog_files) - set(existing_files))
    if missing_files:
        warnings.warn(
            f"{len(missing_files)} V50 sky-cell file(s) are missing; "
            "calibration will use the available cells only.",
            RuntimeWarning,
            stacklevel=2,
        )
    if not existing_files:
        return _unavailable("no V50 sky-cell files were found for the image stars")

    catalog_ra, catalog_dec, catalog_magnitude = _read_v50_files(existing_files)
    if catalog_ra.size == 0:
        return _unavailable("the relevant V50 sky cells contained no readable stars")

    catalog_coords = SkyCoord(catalog_ra * u.deg, catalog_dec * u.deg, frame="icrs")
    catalog_x, catalog_y = image_info.wcs.world_to_pixel(catalog_coords)
    in_or_near_image = (
        np.isfinite(catalog_x)
        & np.isfinite(catalog_y)
        & (catalog_x >= -match_radius_pixels)
        & (catalog_x < image_info.nx + match_radius_pixels)
        & (catalog_y >= -match_radius_pixels)
        & (catalog_y < image_info.ny + match_radius_pixels)
    )
    if not np.any(in_or_near_image):
        return _unavailable("no V50 stars project onto the image")

    pairs = _match_stars(
        stars,
        np.column_stack((catalog_x[in_or_near_image], catalog_y[in_or_near_image])),
        catalog_magnitude[in_or_near_image],
        match_radius_pixels,
    )
    if len(pairs) < _MIN_CALIBRATION_STARS:
        return _unavailable(
            f"only {len(pairs)} image/catalog star matches passed quality checks; "
            f"at least {_MIN_CALIBRATION_STARS} are required"
        )

    matched_rates = np.asarray(
        [stars[image_index].flux / exposure_seconds for image_index, _, _ in pairs],
        dtype=float,
    )
    matched_magnitudes = np.asarray([magnitude for _, _, magnitude in pairs])
    usable = np.isfinite(matched_rates) & (matched_rates > 0) & np.isfinite(matched_magnitudes)
    zero_points = matched_magnitudes[usable] + 2.5 * np.log10(matched_rates[usable])
    if zero_points.size < _MIN_CALIBRATION_STARS:
        return _unavailable(
            f"only {zero_points.size} matches have positive finite flux; "
            f"at least {_MIN_CALIBRATION_STARS} are required"
        )

    # Reject mismatches and bad photometry with a robust median/MAD clipping.
    median = float(np.median(zero_points))
    mad = float(1.4826 * np.median(np.abs(zero_points - median)))
    if mad > 0:
        selected = np.abs(zero_points - median) <= 3.0 * mad
        zero_points = zero_points[selected]
    if zero_points.size < _MIN_CALIBRATION_STARS:
        return _unavailable(
            f"only {zero_points.size} matches remain after robust outlier rejection; "
            f"at least {_MIN_CALIBRATION_STARS} are required"
        )

    value = float(np.median(zero_points))
    scatter = float(1.4826 * np.median(np.abs(zero_points - value)))
    return ZeroPointResult(value, "V50 matched stars", int(zero_points.size), scatter)


def calculate_limiting_magnitude(
    image_info: ImageInfo,
    measurements: ImageMeasurements,
    zero_point: float,
    snr_threshold: float = 3.0,
) -> LimitingMagnitudeResult:
    """Estimate a V limit from sky Poisson noise in the FWHM-radius aperture.

    The measured background is in image units per pixel for the full exposure.
    A recorded gain is used to express Poisson noise back in those same units.
    """
    if image_info.exptime is None:
        return _limiting_magnitude_unavailable("exposure time is missing")
    exposure_seconds = image_info.exptime.to_value(u.s)
    if not np.isfinite(exposure_seconds) or exposure_seconds <= 0:
        return _limiting_magnitude_unavailable("exposure time must be positive")
    if not np.isfinite(zero_point):
        return _limiting_magnitude_unavailable("zero point must be finite")
    if not np.isfinite(snr_threshold) or snr_threshold <= 0:
        return _limiting_magnitude_unavailable("SNR threshold must be positive")
    if (
        measurements.fwhm_pixels is None
        or not np.isfinite(measurements.fwhm_pixels)
        or measurements.fwhm_pixels <= 0
    ):
        return _limiting_magnitude_unavailable("FWHM is unavailable")

    background = np.asarray(measurements.background, dtype=float)
    finite_background = background[np.isfinite(background)]
    if finite_background.size == 0:
        return _limiting_magnitude_unavailable("image background is not finite")
    background_per_pixel = float(np.median(finite_background))
    if background_per_pixel <= 0:
        return _limiting_magnitude_unavailable(
            "median image background is not positive"
        )

    aperture_area = float(np.pi * measurements.fwhm_pixels**2)
    gain = image_info.header_gain_e_per_adu
    gain_correction = gain if gain is not None else 1.0
    sky_variance = aperture_area * background_per_pixel / gain_correction
    sky_noise = float(np.sqrt(sky_variance))
    magnitude = float(
        zero_point
        + 2.5 * np.log10(exposure_seconds / (snr_threshold * sky_noise))
    )
    return LimitingMagnitudeResult(
        magnitude,
        background_per_pixel,
        sky_noise,
        aperture_area,
        gain,
    )


def _unavailable(reason: str) -> ZeroPointResult:
    """Return a clear reason for the caller to include in its status message."""
    return ZeroPointResult(None, "unavailable", unavailable_reason=reason)


def _limiting_magnitude_unavailable(reason: str) -> LimitingMagnitudeResult:
    """Return a limiting-magnitude result with an explicit fallback reason."""
    return LimitingMagnitudeResult(None, unavailable_reason=reason)


def is_v_filter(filter_name: str | None) -> bool:
    """Recognize common explicit names for the Johnson V passband."""
    if filter_name is None:
        return False
    normalized = filter_name.strip().upper().replace("_", " ")
    return normalized in {"V", "JOHNSON V", "JOHNSON-V", "BESSELL V", "V BAND"}


def _catalog_files_for_stars(
    stars: list[ImageStar],
    pixel_scale_arcsec: float,
    match_radius_pixels: float,
    catalog_dir: Path,
) -> list[Path]:
    """Find only V50 sky cells near detected stars and their match radii."""
    radius = match_radius_pixels * pixel_scale_arcsec * u.arcsec
    unique_cells = set()
    for star in stars:
        center = SkyCoord(star.ra_deg * u.deg, star.dec_deg * u.deg, frame="icrs")
        # Query offsets around each star so nearby cells across a boundary are included.
        offsets = center.directional_offset_by(
            np.arange(0, 360, 45) * u.deg,
            radius,
        )
        for ra, dec in zip(
            np.concatenate(([star.ra_deg], offsets.ra.deg)),
            np.concatenate(([star.dec_deg], offsets.dec.deg)),
        ):
            unique_cells.add(_cell_for_coordinate(float(ra), float(dec)))
    return [catalog_dir / f"v50_{ring:02d}{cell:02d}.1476" for ring, cell in sorted(unique_cells)]


def _cell_for_coordinate(ra_deg: float, dec_deg: float) -> tuple[int, int]:
    """Return one-based ASTAP ring/cell indices for an ICRS coordinate."""
    dec = float(np.clip(dec_deg, -90.0, 90.0))
    half_ring = _DEG_PER_RING / 2.0
    if dec < -90.0 + half_ring:
        ring_index = 0
    elif dec >= 90.0 - half_ring:
        ring_index = 35
    else:
        ring_index = int(np.floor((dec + 90.0 - half_ring) / _DEG_PER_RING)) + 1

    if ring_index in (0, 35):
        cell_count = 1
    else:
        dec_min = -90.0 + half_ring + (ring_index - 1) * _DEG_PER_RING
        dec_max = dec_min + _DEG_PER_RING
        outer_boundary = max(abs(dec_min), abs(dec_max))
        cell_count = int(
            np.floor(360.0 * np.cos(np.radians(outer_boundary)) / _DEG_PER_RING)
        )
        cell_count = max(cell_count, 1)

    ra = float(ra_deg) % 360.0
    cell_index = min(int(ra / (360.0 / cell_count)), cell_count - 1)
    return ring_index + 1, cell_index + 1


def _read_v50_files(paths: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode V50 1476-6 records into RA, Dec, and Johnson-V magnitude arrays."""
    ras = []
    decs = []
    magnitudes = []

    for path in paths:
        try:
            data = np.fromfile(path, dtype=np.uint8)
        except OSError as exc:
            warnings.warn(f"Could not read V50 catalog cell {path}: {exc}", RuntimeWarning)
            continue
        if (
            data.size < _CATALOG_HEADER_SIZE
            or (data.size - _CATALOG_HEADER_SIZE) % _V50_RECORD_SIZE != 0
        ):
            warnings.warn(f"Skipping malformed V50 catalog cell: {path}", RuntimeWarning)
            continue

        records = data[_CATALOG_HEADER_SIZE:].reshape((-1, _V50_RECORD_SIZE))
        current_dec9 = None
        current_magnitude = None
        for record in records:
            if np.all(record[:3] == 255):
                # Special records set the declination high byte and magnitude
                # for the following stars in this 0.1-mag/declination group.
                current_dec9 = int(record[3]) - 128
                current_magnitude = (int(record[4]) - 16) / 10.0
                continue
            if current_dec9 is None or current_magnitude is None:
                continue

            ra_integer = (
                int(record[0])
                + (int(record[1]) << 8)
                + (int(record[2]) << 16)
            )
            dec_integer = (
                int(record[3])
                + (int(record[4]) << 8)
                + (current_dec9 << 16)
            )
            ras.append(ra_integer * 360.0 / ((1 << 24) - 1))
            decs.append(dec_integer * 90.0 / ((1 << 23) - 1))
            magnitudes.append(current_magnitude)

    return (
        np.asarray(ras, dtype=float),
        np.asarray(decs, dtype=float),
        np.asarray(magnitudes, dtype=float),
    )


def _match_stars(
    image_stars: list[ImageStar],
    catalog_positions: np.ndarray,
    catalog_magnitudes: np.ndarray,
    radius_pixels: float,
) -> list[tuple[int, int, float]]:
    """Pair stars one-to-one by nearest WCS-projected pixel positions."""
    if not image_stars or catalog_positions.size == 0:
        return []
    image_positions = np.asarray([(star.x, star.y) for star in image_stars])
    distances, image_indices = cKDTree(image_positions).query(catalog_positions, k=1)
    candidates = [
        (float(distance), int(image_index), catalog_index)
        for catalog_index, (distance, image_index) in enumerate(zip(distances, image_indices))
        if distance <= radius_pixels
    ]

    # Greedy nearest-first assignment prevents one image detection matching multiple catalog stars.
    matched_image = set()
    matched_catalog = set()
    pairs = []
    for _, image_index, catalog_index in sorted(candidates):
        if image_index in matched_image or catalog_index in matched_catalog:
            continue
        matched_image.add(image_index)
        matched_catalog.add(catalog_index)
        pairs.append((image_index, catalog_index, float(catalog_magnitudes[catalog_index])))
    return pairs
