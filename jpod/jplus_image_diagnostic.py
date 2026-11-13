"""Explore FWHM and sky-limited depth for a J-PLUS image.

Run from the repository root with:
    python -m jpod.jplus_image_diagnostic [path/to/image.fz]

The reported 3-arcsec-aperture magnitude is a sky-Poisson-only estimate for
comparison with the J-PLUS DR3 5-sigma, 3-arcsec-aperture reference. It uses
the image-header ZP and can treat pixel values as exposure-normalized rates
or integrated counts.
"""

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

from . import image_analysis
from .image_geometry import load_image

_DEFAULT_IMAGE = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "j-plus:rSDSS"
    / "j02-20240314T010726-01_v202505.fz"
)

_PROFILE_SETTINGS = (
    ("candidates=50", {"_MAX_PROFILE_CANDIDATES": 50}),
    ("candidates=100", {"_MAX_PROFILE_CANDIDATES": 100}),
    ("candidates=200", {"_MAX_PROFILE_CANDIDATES": 200}),
    ("candidates=500", {"_MAX_PROFILE_CANDIDATES": 500}),
    ("candidates=1000", {"_MAX_PROFILE_CANDIDATES": 1000}),
    ("candidates=2000", {"_MAX_PROFILE_CANDIDATES": 2000}),
    ("candidates=all", {"_MAX_PROFILE_CANDIDATES": None}),
    (
        "all, peak filter=5",
        {"_MAX_PROFILE_CANDIDATES": None, "_PEAK_FILTER_SIZE": 5},
    ),
    (
        "all, peak filter=13",
        {"_MAX_PROFILE_CANDIDATES": None, "_PEAK_FILTER_SIZE": 13},
    ),
    (
        "all, min SNR=10",
        {"_MAX_PROFILE_CANDIDATES": None, "_MIN_PROFILE_SNR": 10.0},
    ),
    (
        "all, min SNR=30",
        {"_MAX_PROFILE_CANDIDATES": None, "_MIN_PROFILE_SNR": 30.0},
    ),
    (
        "all, elongation=1.5",
        {"_MAX_PROFILE_CANDIDATES": None, "_MAX_ELONGATION": 1.5},
    ),
    (
        "all, elongation=1.2",
        {"_MAX_PROFILE_CANDIDATES": None, "_MAX_ELONGATION": 1.2},
    ),
    (
        "all, profile half-size=7",
        {"_MAX_PROFILE_CANDIDATES": None, "_PROFILE_HALF_SIZE": 7},
    ),
)


def _read_image_header(path: Path) -> fits.Header:
    """Return the header belonging to the science image HDU."""
    with fits.open(path) as hdul:
        image_hdu = hdul[0] if hdul[0].header.get("NAXIS", 0) >= 2 else None
        if image_hdu is None:
            image_hdu = next(
                (
                    hdu
                    for hdu in hdul[1:]
                    if hdu.is_image
                    and hdu.header.get("NAXIS", 0) >= 2
                    and str(hdu.header.get("EXTNAME", "")).strip().upper()
                    == "IMAGE"
                ),
                None,
            )
        if image_hdu is None:
            raise ValueError(f"No science image HDU found in {path}")
        return image_hdu.header.copy()


def _sky_limited_magnitude(
    zero_point: float,
    background_per_pixel: float,
    gain_e_per_adu: float,
    exposure_seconds: float,
    aperture_area_pixels: float,
    pixel_values_are_rates: bool,
) -> float:
    """Estimate a 5-sigma magnitude from sky Poisson noise only."""
    if exposure_seconds <= 0 or aperture_area_pixels <= 0:
        raise ValueError("Exposure time and aperture area must be positive.")
    if background_per_pixel <= 0 or gain_e_per_adu <= 0:
        raise ValueError("Sky background and gain must be positive.")

    time_power = 1 if pixel_values_are_rates else 2
    sky_variance_rate = (
        aperture_area_pixels
        * background_per_pixel
        / (gain_e_per_adu * exposure_seconds**time_power)
    )
    five_sigma_rate = 5.0 * np.sqrt(sky_variance_rate)
    return float(zero_point - 2.5 * np.log10(five_sigma_rate))


def _measure_with_settings(image_info, settings, background_cache):
    """Run the production FWHM estimator with temporary diagnostic settings."""
    original_settings = {
        name: getattr(image_analysis, name)
        for name in settings
    }
    original_background2d = image_analysis.Background2D

    def cached_background2d(*args, **kwargs):
        if background_cache["background"] is None:
            result = original_background2d(*args, **kwargs)
            background_cache["background"] = result.background
            return result
        return SimpleNamespace(background=background_cache["background"])

    try:
        image_analysis.Background2D = cached_background2d
        for name, value in settings.items():
            setattr(image_analysis, name, value)
        return image_analysis.measure_image(image_info)
    finally:
        image_analysis.Background2D = original_background2d
        for name, value in original_settings.items():
            setattr(image_analysis, name, value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "image",
        nargs="?",
        type=Path,
        default=_DEFAULT_IMAGE,
        help=f"FITS or .fz image (default: {_DEFAULT_IMAGE})",
    )
    parser.add_argument(
        "--exposure-seconds",
        nargs="+",
        type=float,
        default=[1.0, 19.93, 20.431],
        metavar="SECONDS",
        help=(
            "Physical exposure-time assumptions for the sky-noise estimate "
            "(default: 1, TEXPTIME, DATE-END minus DATE-OBS)."
        ),
    )
    parser.add_argument(
        "--pixel-values",
        choices=("rates", "integrated"),
        default="rates",
        help=(
            "Interpret pixels as exposure-normalized ADU/s (default) or "
            "integrated ADU counts."
        ),
    )
    args = parser.parse_args()

    image_path = args.image.expanduser().resolve()
    if not image_path.is_file():
        parser.error(f"image does not exist: {image_path}")
    if any(not np.isfinite(t) or t <= 0 for t in args.exposure_seconds):
        parser.error("all exposure times must be positive and finite")

    header = _read_image_header(image_path)
    if "ZP" not in header:
        parser.error("the image header has no ZP card")
    if "GAIN" not in header:
        parser.error("the image header has no GAIN card")

    obscodes_path = Path(__file__).with_name("fits_header") / "obscodes_extended.json"
    info = load_image(str(image_path), obscodes_json=str(obscodes_path))
    pixel_scale = float(proj_plane_pixel_scales(WCS(header).celestial)[0] * 3600)
    header_fwhm_arcsec = header.get("OAJ PRO FWHMSEXT")
    header_fwhm_pixels = (
        float(header_fwhm_arcsec) / pixel_scale
        if header_fwhm_arcsec is not None
        else None
    )
    zero_point = float(header["ZP"])
    gain = float(header["GAIN"])
    filter_name = str(header.get("FILTER", "unknown")).strip()
    exposure_span = None
    if "DATE-OBS" in header and "DATE-END" in header:
        from astropy.time import Time

        exposure_span = float(
            (Time(header["DATE-END"]) - Time(header["DATE-OBS"])).sec
        )

    print(f"Image: {image_path}")
    print(f"Filter: {filter_name}; BUNIT={header.get('BUNIT', 'unspecified')}")
    print(f"Pixel scale: {pixel_scale:.4f} arcsec/pixel")
    print(f"Header ZP: {zero_point:.3f}; gain: {gain:.4g} e-/ADU")
    if header_fwhm_pixels is not None:
        print(
            f"Header OAJ PRO FWHMSEXT: {float(header_fwhm_arcsec):.3f} arcsec "
            f"= {header_fwhm_pixels:.3f} pixels"
        )
    else:
        print("Header OAJ PRO FWHMSEXT: unavailable")
    if info.exptime is not None:
        print(f"Parsed exposure keyword: {info.exptime.to_value('s'):.3f} s")
    if exposure_span is not None:
        print(f"DATE-END - DATE-OBS: {exposure_span:.3f} s")

    data = np.asarray(info.data, dtype=float)
    finite_data = data[np.isfinite(data)]
    if finite_data.size == 0:
        parser.error("image has no finite pixel values")
    background_cache = {"background": None}

    exposure_labels = []
    for exposure in args.exposure_seconds:
        if np.isclose(exposure, 1.0):
            label = "1s"
        elif "TEXPTIME" in header and np.isclose(
            exposure, float(header["TEXPTIME"])
        ):
            label = "TEXPTIME"
        elif exposure_span is not None and np.isclose(exposure, exposure_span):
            label = "DATE span"
        else:
            label = f"{exposure:g}s"
        exposure_labels.append((label, exposure))

    print(
        "\nFWHM parameter sweep; magnitudes are sky-Poisson-only, "
        "5 sigma, using the header ZP."
    )
    print(
        "The 3-arcsec reference aperture is a 3-arcsec diameter; "
        "the j-pod-style aperture has radius equal to measured FWHM."
    )

    for label, settings in _PROFILE_SETTINGS:
        measurements = _measure_with_settings(info, settings, background_cache)
        fwhm = measurements.fwhm_pixels
        if fwhm is None:
            print(f"\n{label}: FWHM unavailable")
            continue

        background = np.asarray(measurements.background, dtype=float)
        finite_background = background[np.isfinite(background)]
        if finite_background.size == 0:
            print(f"\n{label}: no finite background estimate")
            continue
        background_per_pixel = float(np.median(finite_background))
        area_3_arcsec = np.pi * (1.5 / pixel_scale) ** 2
        area_fwhm = np.pi * fwhm**2

        print(
            f"\n{label}: FWHM={fwhm:.3f} px "
            f"({fwhm * pixel_scale:.3f} arcsec), "
            f"calibration stars retained={len(measurements.stars)}, "
            f"median sky={background_per_pixel:.4g} pixel-unit/pixel"
        )
        reference_values = []
        fwhm_values = []
        for exposure_label, exposure in exposure_labels:
            reference = _sky_limited_magnitude(
                zero_point,
                background_per_pixel,
                gain,
                exposure,
                area_3_arcsec,
                args.pixel_values == "rates",
            )
            fwhm_limit = _sky_limited_magnitude(
                zero_point,
                background_per_pixel,
                gain,
                exposure,
                area_fwhm,
                args.pixel_values == "rates",
            )
            reference_values.append(f"{exposure_label}: {reference:.2f}")
            fwhm_values.append(f"{exposure_label}: {fwhm_limit:.2f}")
        print(f"  5-sigma, 3-arcsec diameter: {', '.join(reference_values)}")
        print(f"  5-sigma, FWHM-radius:      {', '.join(fwhm_values)}")

    print(
        "\nInterpretation: compare the 3-arcsec-diameter result with the "
        "paper's approximate 21.8-mag DR3 survey depth. It is not expected "
        "to match exactly: this calculation uses one image's median sky, "
        "the header ZP, and sky Poisson noise only."
    )
    print(
        f"Pixel model used: {'exposure-normalized rate' if args.pixel_values == 'rates' else 'integrated counts'}."
    )


if __name__ == "__main__":
    main()
