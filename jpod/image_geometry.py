"""Image loading and geometry: WCS, orientation, ellipse angles, overlap filter."""

from dataclasses import dataclass
import re
from typing import Literal

import astropy.units as u
from astropy.table import Column
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from .fits_header.fits_header import get_obs_times, get_observatory_code
from .fits_header.obscodes import load_obscodes_safe

ImageUnitKind = Literal[
    "adu",
    "electron",
    "adu_per_second",
    "electron_per_second",
    "unknown",
]


@dataclass
class ImageInfo:
    """Everything we need to know about the FITS image."""
    data: np.ndarray
    wcs: WCS
    nx: int
    ny: int
    center: SkyCoord          # sky position of the image center
    corners: SkyCoord         # sky positions of the 4 corners
    pixel_scale: float        # arcsec/pixel (square pixels assumed)
    observatory_code: str     # MPC observatory code
    obs_code_source: str      # where the code was found
    date_obs: Time            # time written in the header
    start_obs: Time           # start of the exposure (UTC)
    end_obs: Time             # end of the exposure (UTC)
    exptime: object           # exposure time (Quantity) or None
    time_source: str          # where the time was found
    header_fwhm: float | None # FITS FWHM, assumed to be in pixels
    header_zero_points: tuple[tuple[str, float], ...]
    header_saturation: float | None
    header_gain_e_per_adu: float | None
    filter_name: str | None
    image_unit_kind: ImageUnitKind = "unknown"
    bunit: str | None = None
    image_unit_source: str | None = None
    image_unit_reason: str | None = None


def load_image(fits_file: str, observatory_code = None, time_reference = None, date_obs = None, obscodes_json = None)-> ImageInfo:
    """Read the FITS file and return an ImageInfo.
    observatory_code : fallback code used only if neither the header nor
        obscodes_json provides a code; otherwise it defaults to geocentric 500
    time_reference : None (automatic) or 'start', 'mid', 'end' to force what the time in the header means
    obscodes_json: path of the MPC JSON list of observatory codes
    date_obs: force the observation time (UTC string); if None it is taken from the header
    """
    with fits.open(fits_file) as hdul:
        image_hdu = hdul[0] if hdul[0].header.get("NAXIS", 0) >= 2 else None
        if image_hdu is None:
            image_hdus = [
                hdu for hdu in hdul[1:]
                if hdu.is_image and hdu.header.get("NAXIS", 0) >= 2
            ]
            image_hdu = next(
                (
                    hdu for hdu in image_hdus
                    if str(hdu.header.get("EXTNAME", "")).strip().upper() == "IMAGE"
                ),
                image_hdus[0] if image_hdus else None,
            )
        if image_hdu is None:
            raise ValueError(f"No image HDU found in FITS file: {fits_file}")

        header = image_hdu.header.copy()
        image_data = image_hdu.data
        if image_data is None:
            raise ValueError(f"Image HDU contains no data in FITS file: {fits_file}")
        data = image_data.copy()

    nx, ny = header["NAXIS1"], header["NAXIS2"]
    w = WCS(header)

    center = SkyCoord.from_pixel((nx - 1) / 2, (ny - 1) / 2, w)
    corners = SkyCoord.from_pixel(
        np.array([0, nx - 1, nx - 1, 0]),
        np.array([0, 0, ny - 1, ny - 1]),
        w,
    )
    obscodes = load_obscodes_safe(obscodes_json)
    code, code_source = get_observatory_code(header, observatory_code, obscodes)
    date_obs, start_obs, end_obs, exptime, time_source = get_obs_times(header, time_reference, date_obs)
    header_fwhm = header.get("FWHM")
    if header_fwhm is not None:
        header_fwhm = float(header_fwhm)

    # Keep supported zero-point cards so calibration can apply precedence
    # without reopening the FITS file.
    header_zero_points = []
    for keyword in ("ZP", "MAGZERO", "PHOTZP"):
        if keyword not in header:
            continue
        try:
            value = float(header[keyword])
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            header_zero_points.append((keyword, value))

    # A saturation threshold lets aperture photometry reject clipped stars.
    header_saturation = None
    for keyword in ("SATURATE", "SATLEVEL"):
        if keyword not in header:
            continue
        try:
            value = float(header[keyword])
        except (TypeError, ValueError):
            continue
        if np.isfinite(value) and value > 0:
            header_saturation = value
            break

    # FITS gain values convert data numbers (ADU/DN) to photoelectrons.
    header_gain_e_per_adu = None
    for keyword in ("EGAIN", "GAIN"):
        if keyword not in header:
            continue
        try:
            value = float(header[keyword])
        except (TypeError, ValueError):
            continue
        if np.isfinite(value) and value > 0:
            header_gain_e_per_adu = value
            break

    image_unit_kind, bunit, unit_source, unit_reason = _classify_image_units(header)

    filter_name = header.get("FILTER")
    if filter_name is not None:
        filter_name = str(filter_name).strip() or None

    scale = proj_plane_pixel_scales(w.celestial)[0] * 3600

    return ImageInfo(data, w, nx, ny, center, corners, scale,
                     code, code_source, date_obs, start_obs, end_obs, exptime,
                     time_source, header_fwhm, tuple(header_zero_points),
                     header_saturation, header_gain_e_per_adu, filter_name,
                     image_unit_kind, bunit, unit_source, unit_reason)


def _classify_image_units(header) -> tuple[ImageUnitKind, str | None, str | None, str]:
    """Classify image units from explicit metadata or ADU-specific header clues."""
    bunit_value = header.get("BUNIT")
    bunit = str(bunit_value).strip() if bunit_value is not None else None
    if not bunit:
        bunit = None

    clues: list[tuple[str, str]] = []
    if bunit:
        clues.append(("BUNIT", bunit))
        try:
            bunit_comment = header.cards["BUNIT"].comment.strip()
        except (KeyError, AttributeError):
            bunit_comment = ""
        if bunit_comment:
            clues.append(("BUNIT comment", bunit_comment))

    related_keys = {"BTYPE", "DATATYPE", "IMAGEUNIT", "UNITS"}
    for card in header.cards:
        key = card.keyword.upper()
        comment = card.comment.strip()
        if key in related_keys or (key.startswith("BUNIT") and key != "BUNIT"):
            clue = str(card.value).strip()
            if clue:
                clues.append((key, clue))
            if comment:
                clues.append((f"{key} comment", comment))
        elif key in {"COMMENT", "HISTORY"}:
            text = str(card.value).strip()
            if _is_image_unit_description(text):
                clues.append((key, text))
        elif comment and _is_image_unit_description(comment):
            clues.append((f"{key} comment", comment))

    candidates: list[tuple[ImageUnitKind, str]] = []
    for source, clue in clues:
        classification = _classify_unit_clue(clue)
        if classification is not None:
            candidates.append((classification, source))

    distinct_kinds = {kind for kind, _ in candidates}
    rate_sources = [
        source for source, clue in clues if _is_rate_unit_clue(clue)
    ]
    if len(distinct_kinds) == 1:
        kind = next(iter(distinct_kinds))
        if kind in {"adu", "electron"} and rate_sources:
            sources = ", ".join(
                dict.fromkeys(
                    [source for _, source in candidates] + rate_sources
                )
            )
            return (
                "unknown",
                bunit,
                sources,
                "unit metadata conflicts about whether image values are rates",
            )
        sources = ", ".join(dict.fromkeys(source for _, source in candidates))
        return kind, bunit, sources, f"identified as {kind.replace('_', ' ')}"
    if len(distinct_kinds) > 1:
        sources = ", ".join(dict.fromkeys(source for _, source in candidates))
        return (
            "unknown",
            bunit,
            sources,
            "unit metadata conflicts; image units are ambiguous",
        )

    adu_evidence = _adu_header_evidence(header)
    if adu_evidence:
        if rate_sources:
            return (
                "unknown",
                bunit,
                ", ".join(dict.fromkeys(adu_evidence + rate_sources)),
                "ADU/DN clues do not resolve whether the image values are rates",
            )
        sources = ", ".join(adu_evidence)
        return (
            "adu",
            bunit,
            sources,
            "inferred as integrated ADU/DN from ADU-specific header comments",
        )

    if not clues:
        reason = "BUNIT and explicit unit-related comments are absent"
    elif any(re.search(r"\bcounts?\b", clue, re.IGNORECASE) for _, clue in clues):
        reason = "generic count units do not distinguish ADU/DN from electrons"
    else:
        reason = "unit metadata does not identify a supported image unit"
    return "unknown", bunit, None, reason


def _adu_header_evidence(header) -> list[str]:
    """Return recognized ADU-specific clues, excluding gain comments alone."""
    image_unit_evidence = []
    for key in ("PEDESTAL", "CBLACK", "CWHITE"):
        if key not in header:
            continue
        comment = header.cards[key].comment.strip()
        if re.search(r"\b(?:adus?|dns?)\b", comment, re.IGNORECASE):
            image_unit_evidence.append(f"{key} comment")
    if not image_unit_evidence:
        return []

    evidence = image_unit_evidence.copy()
    for key in ("EGAIN", "GAIN"):
        if key not in header:
            continue
        comment = header.cards[key].comment.strip()
        if re.search(
            r"(?:\belectrons?\b|\be-\s*)\s*(?:/|per\s+)\s*\b(?:adus?|dns?)\b",
            comment,
            re.IGNORECASE,
        ):
            evidence.append(f"{key} gain-unit comment")
    return evidence


def _is_image_unit_description(text: str) -> bool:
    unit = r"(?:\b(?:adus?|dns?|electrons?|counts?)\b|(?<![a-z])e-(?=\s|/|$))"
    description = r"\b(?:image|pixel|data)\b.{0,40}" + unit
    reverse = unit + r".{0,40}\b(?:image|pixel|data)\b"
    return bool(re.search(f"(?:{description}|{reverse})", text, re.IGNORECASE))


def _classify_unit_clue(clue: str) -> ImageUnitKind | None:
    """Recognize only explicit ADU/DN or electron units and second-based rates."""
    normalized = clue.lower().replace("−", "-").replace("⁻¹", "^-1")
    is_rate = _is_rate_unit_clue(normalized)
    is_electron = bool(
        re.search(r"\b(?:electrons?|e-)\b|(?<![a-z])e-(?=\s|/|$)", normalized)
    )
    is_adu = bool(
        re.search(r"\b(?:adus?|dns?)\b|\bdigital\s+numbers?\b", normalized)
    )
    if is_electron == is_adu:
        return None
    if is_electron:
        return "electron_per_second" if is_rate else "electron"
    return "adu_per_second" if is_rate else "adu"


def _is_rate_unit_clue(clue: str) -> bool:
    normalized = clue.lower().replace("−", "-").replace("⁻¹", "^-1")
    return bool(
        re.search(
            r"(?:/|per\s*)(?:s|sec(?:ond)?s?)\b"
            r"|(?:^|[\s*])(?:s|sec(?:ond)?)\s*(?:\^|\*\*)?\s*-1(?:\b|$)",
            normalized,
        )
    )


def describe_time(info: ImageInfo)->list:
    """Lines describing the observation time, for the terminal."""
    lines = [f"Time in header: {info.date_obs.isot} UTC ({info.time_source})"]
    if info.exptime is not None:
        lines.append(f"Exposure: {info.exptime:.1f}, from {info.start_obs.isot} to {info.end_obs.isot} UTC.")
    return lines


def search_radius(info: ImageInfo, margin=0 * u.arcmin):
    """Radius of the circle containing the whole image, plus a margin."""
    distances = info.center.separation(info.corners)
    return np.max(distances).deg * u.deg + margin


def sky_direction_angle(w: WCS, coords, position_angle, step=10 * u.arcsec):
    """Angle in the image of a sky direction, at one or many positions.

    position_angle is measured from North through East (0 = N, 90 = E).
    The result is in degrees, counterclockwise from the +x axis
    (0 = right, 90 = up), assuming y increases upward in the image.
    """
    x0, y0 = w.world_to_pixel(coords)                             # starting pixel
    shifted = coords.directional_offset_by(position_angle, step)  # small step on the sky
    x1, y1 = w.world_to_pixel(shifted)                            # back to pixels
    return np.degrees(np.arctan2(y1 - y0, x1 - x0)) % 360.0


def image_orientation(w: WCS, coords):
    """Angle of North and East in the image, and the parity of the image.

    parity = +1 if East is counterclockwise from North, -1 if mirrored.
    """
    angle_north = sky_direction_angle(w, coords, 0 * u.deg)
    angle_east = sky_direction_angle(w, coords, 90 * u.deg)
    parity = 1 if ((angle_east - angle_north) % 360.0) < 180.0 else -1
    return angle_north, angle_east, parity


def add_ellipse_image_angle(table, w: WCS):
    """Add 'ellipse_angle': orientation of the uncertainty ellipse's major
    axis in the image (deg, CCW from +x, 0 = right, 90 = up), in [0, 180)."""
    angles = np.full(len(table), np.nan)

    pa = u.Quantity(table["LOV_PA"]).to(u.deg)
    ok = np.isfinite(pa.value)
    if ok.any():
        coords = SkyCoord(
            u.Quantity(table["RA"]).to(u.deg)[ok],
            u.Quantity(table["DEC"]).to(u.deg)[ok],
            frame="icrs",
        )
        # An ellipse is symmetric under a 180 deg rotation.
        angles[ok] = sky_direction_angle(w, coords, pa[ok]) % 180.0

    table["ellipse_angle"] = angles * u.deg


def ellipse_overlaps_image(x, y, a, b, angle_deg, nx, ny, n_samples=720)-> bool:
    """True if the ellipse (center x,y; semi-axes a,b in pixels; major axis at
    angle_deg CCW from +x) overlaps the image rectangle."""
    xmin, xmax, ymin, ymax = -0.5, nx - 0.5, -0.5, ny - 0.5

    def inside(px, py):
        return (px >= xmin) & (px <= xmax) & (py >= ymin) & (py <= ymax)

    # 1. center inside the image
    if inside(x, y):
        return True

    th = np.radians(angle_deg)
    c, s = np.cos(th), np.sin(th)

    # 2. some point of the outline inside the image
    t = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    ex = x + a * np.cos(t) * c - b * np.sin(t) * s
    ey = y + a * np.cos(t) * s + b * np.sin(t) * c
    if np.any(inside(ex, ey)):
        return True

    # 3. ellipse containing the image: some corner inside the ellipse
    cx = np.array([xmin, xmax, xmax, xmin]) - x
    cy = np.array([ymin, ymin, ymax, ymax]) - y
    u_ = cx * c + cy * s
    v_ = -cx * s + cy * c
    return bool(np.any((u_ / a) ** 2 + (v_ / b) ** 2 <= 1))


def filter_in_image(table, info: ImageInfo):
    """Keep objects whose position or uncertainty ellipse overlaps the image.
    Adds the columns 'x_pix' and 'y_pix' (pixel position of each object at the start of the exposure)
    and 'x_pix_end', 'y_pix_end'. Without the columns RA_end, DEC_end the two positions are the same.
    """
    nx, ny, scale = info.nx, info.ny, info.pixel_scale
    def to_pixel(ra_name, dec_name):
        coords = SkyCoord(u.Quantity(table[ra_name]).to(u.deg),
                          u.Quantity(table[dec_name]).to(u.deg), frame = "icrs")
        return info.wcs.world_to_pixel(coords)
    
    x, y = to_pixel("RA", "DEC")
    if "RA_end" in table.colnames:
        x_end, y_end = to_pixel("RA_end", "DEC_end")
    else:
        x_end, y_end = x, y

    a = u.Quantity(table["SMAA_3sigma"]).to_value(u.arcsec) / scale
    b = u.Quantity(table["SMIA_3sigma"]).to_value(u.arcsec) / scale
    ang = u.Quantity(table["ellipse_angle"]).to_value(u.deg)
    p = u.Quantity(table["posunc"]).to_value(u.arcsec) / scale

    def overlaps(i, xi, yi):
        if np.all(np.isfinite([a[i], b[i], ang[i]])):
            return ellipse_overlaps_image(xi, yi, a[i], b[i], ang[i], nx, ny)
        # Without a Horizons ellipse, use the provider's positional uncertainty.
        r = max(p[i] if np.isfinite(p[i]) else 0.0, 1e-6)
        return ellipse_overlaps_image(xi, yi, r, r, 0.0, nx, ny)

    keep = np.array([overlaps(i,x[i],y[i]) or overlaps(i, x_end[i], y_end[i])
                    for i in range(len(table))], dtype=bool)
    
    table["x_pix"] = x
    table["y_pix"] = y
    table["x_pix_end"] = x_end
    table["y_pix_end"] = y_end
    return table[keep]

def compute_trail_length(info: ImageInfo, table):
    """Compute the length (in pixels and arcseconds) of the trail produced by an object during the exposure. 
    It can be computed in two ways: 
    1) by using the starting and ending pixels, computing their distance and converting it to arcseconds using pixel_scale
    2) by multiplying sky_speed * exp_time and converting it to arcseconds and then to pixels. 
    We prefer the second way. 
    """
    if info.exptime is None:
        table["trail_length"] = np.full(len(table), np.nan) * u.arcsec
        _set_pixel_trail_column(table, np.full(len(table), np.nan))
        return

    sky_speed = table["sky_speed"] # already in arcsec/hour
    exptime = info.exptime.to(u.s) 
    pixel_scale = info.pixel_scale * u.arcsec / u.pixel
    trail_length = (sky_speed * exptime).to(u.arcsec)
    table["trail_length"] = trail_length
    pixel_trail = (trail_length / pixel_scale).to_value(u.pixel)
    _set_pixel_trail_column(table, pixel_trail)


def _set_pixel_trail_column(table, values):
    """Set the plain numeric pixel column with an explicit Astropy name."""
    column = Column(np.asarray(values), name="trail_length [px]")
    if "trail_length [px]" in table.colnames:
        table.replace_column("trail_length [px]", column)
    else:
        table.add_column(column)
