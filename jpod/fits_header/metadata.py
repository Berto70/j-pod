"""Resolve FITS header conventions into normalized image metadata."""

from dataclasses import dataclass
import math
import re
from typing import Literal

import astropy.units as u
from astropy.time import Time

from .obscodes import find_code

ImageUnitKind = Literal[
    "adu",
    "electron",
    "adu_per_second",
    "electron_per_second",
    "unknown",
]


@dataclass(frozen=True)
class HeaderProfile:
    """Known header-card conventions for a family of image products."""

    name: str
    signature_cards: tuple[str, ...] = ()
    longitude_cards: tuple[str, ...] = ("SITELON", "OBSLON", "LONGITUD")
    latitude_cards: tuple[str, ...] = ("SITELAT", "OBSLAT", "LATITUDE")
    altitude_cards: tuple[str, ...] = ("OBSALT", "ALTITUDE")
    fwhm_cards: tuple[tuple[str, str], ...] = (("FWHM", "pixel"),)
    texptime_cards: tuple[str, ...] = ()
    trail_duration_preference: Literal[
        "texptime", "observation_interval"
    ] = "observation_interval"


GENERIC_PROFILE = HeaderProfile("generic")
OAJ_PROFILE = HeaderProfile(
    "OAJ",
    signature_cards=("OAJ TEL GEOLON", "OAJ PRO FWHMSEXT"),
    longitude_cards=GENERIC_PROFILE.longitude_cards + ("OAJ TEL GEOLON",),
    latitude_cards=GENERIC_PROFILE.latitude_cards + ("OAJ TEL GEOLAT",),
    altitude_cards=GENERIC_PROFILE.altitude_cards + ("OAJ TEL GEOELEV",),
    fwhm_cards=(
        ("OAJ PRO FWHMSEXT", "arcsec"),
        ("FWHM", "pixel"),
    ),
    texptime_cards=("TEXPTIME",),
    trail_duration_preference="texptime",
)
HEADER_PROFILES = (OAJ_PROFILE, GENERIC_PROFILE)

TIME_KEYWORDS = (
    ("DATE-AVG", "iso", "mid"),
    ("MJD-AVG", "mjd", "mid"),
    ("DATE-BEG", "iso", "start"),
    ("MJD-BEG", "mjd", "start"),
    ("DATE-OBS", "iso", None),
    ("OBSJD", "jd", None),
    ("OBSMJD", "mjd", None),
    ("MJD-OBS", "mjd", None),
    ("SHUTOPEN", "iso", "start"),
)
END_TIME_KEYWORDS = (("DATE-END", "iso"), ("MJD-END", "mjd"))
EXPTIME_KEYWORDS = ("EXPTIME", "EXPOSURE", "EXPTIM")
OBS_CODE_KEYWORDS = ("OBSCODE", "MPCCODE", "OBS-CODE")
GEOCENTRIC_CODE = "500"
OBS_CODE_PATTERN = re.compile(r"^(\d{3}|[A-Za-z]\d{2})$")
TIME_REFERENCES = {"start": 1.0, "mid": 0.5, "end": 0.0}
COMMENT_PATTERNS = {
    "start": r"\b(start\w*|begin\w*|open\w*)\b",
    "mid": r"\b(mid\w*|middle|average|avg|cent(?:er|re))\b",
    "end": r"\b(end\w*|stop\w*|close\w*)\b",
}


@dataclass(frozen=True)
class HeaderMetadata:
    """Normalized header values with the source information needed to audit them."""

    profile_name: str
    observatory_code: str
    obs_code_source: str
    date_obs: Time
    start_obs: Time
    end_obs: Time
    exptime: u.Quantity | None
    texptime: u.Quantity | None
    texptime_source: str | None
    time_source: str
    observation_duration: u.Quantity | None
    trail_duration: u.Quantity | None
    trail_duration_source: str | None
    header_fwhm_pixels: float | None
    header_fwhm_value: float | None
    header_fwhm_unit: str | None
    header_fwhm_source: str | None
    header_lim_mag: float | None
    header_zero_points: tuple[tuple[str, float], ...]
    header_saturation: float | None
    header_gain_e_per_adu: float | None
    filter_name: str | None
    image_unit_kind: ImageUnitKind
    bunit: str | None
    image_unit_source: str | None
    image_unit_reason: str


def select_header_profile(header) -> HeaderProfile:
    """Select a profile only when a distinctive instrument card is present."""
    for profile in HEADER_PROFILES:
        if any(key in header for key in profile.signature_cards):
            return profile
    return GENERIC_PROFILE


def resolve_header_metadata(
    header,
    pixel_scale_arcsec: float,
    observatory_code=None,
    obscodes=None,
    time_reference=None,
    date_obs_override=None,
) -> HeaderMetadata:
    """Interpret one already-open image HDU header without reopening its file."""
    profile = select_header_profile(header)
    code, code_source = _get_observatory_code(
        header, profile, observatory_code, obscodes
    )
    (
        date_obs,
        start_obs,
        end_obs,
        exptime,
        time_source,
        observation_duration,
    ) = _get_observation_times(header, time_reference, date_obs_override)
    texptime, texptime_source = _get_profile_exposure_time(header, profile)
    _, end_key = (
        _get_header_end_time(header)
        if date_obs_override is None
        else (None, None)
    )
    trail_duration, trail_duration_source = _select_trail_duration(
        profile,
        texptime,
        texptime_source,
        observation_duration,
        end_key,
        exptime,
    )

    fwhm_value, fwhm_unit, fwhm_source = _get_header_fwhm(header, profile)
    if fwhm_value is None:
        fwhm_pixels = None
    elif fwhm_unit == "arcsec":
        fwhm_pixels = fwhm_value / pixel_scale_arcsec
    else:
        fwhm_pixels = fwhm_value

    header_lim_mag = header["MAGLIM"] if "MAGLIM" in header else None

    header_zero_points = []
    for keyword in ("ZP", "MAGZERO", "PHOTZP"):
        if keyword not in header:
            continue
        try:
            value = float(header[keyword])
        except (TypeError, ValueError):
            continue
        if _is_finite(value):
            header_zero_points.append((keyword, value))

    header_saturation = _first_positive_number(header, ("SATURATE", "SATLEVEL"))
    header_gain = _first_positive_number(header, ("EGAIN", "GAIN"))
    image_unit_kind, bunit, unit_source, unit_reason = _classify_image_units(header)
    filter_name = header.get("FILTER")
    if filter_name is not None:
        filter_name = str(filter_name).strip() or None

    return HeaderMetadata(
        profile.name,
        code,
        code_source,
        date_obs,
        start_obs,
        end_obs,
        exptime,
        texptime,
        texptime_source,
        time_source,
        observation_duration,
        trail_duration,
        trail_duration_source,
        fwhm_pixels,
        fwhm_value,
        fwhm_unit,
        fwhm_source,
        header_lim_mag,
        tuple(header_zero_points),
        header_saturation,
        header_gain,
        filter_name,
        image_unit_kind,
        bunit,
        unit_source,
        unit_reason,
    )


def get_observatory_code(header, fallback_code=None, obscodes=None):
    """Compatibility helper for callers that only need the resolved site."""
    profile = select_header_profile(header)
    return _get_observatory_code(header, profile, fallback_code, obscodes)


def get_header_time(header, override=None):
    """Return (time, keyword, semantic reference, comment) from the header."""
    if override:
        return Time(str(override).strip(), scale="utc"), "set manually", None, ""
    for key, kind, meaning in TIME_KEYWORDS:
        if key not in header:
            continue
        try:
            value = header[key]
            if key == "DATE-OBS" and "T" not in str(value) and "TIME-OBS" in header:
                value = f"{str(value).strip()}T{str(header['TIME-OBS']).strip()}"
            return _read_time(value, kind), key, meaning, _comment(header, key)
        except (ValueError, TypeError):
            continue
    tried = [key for key, _, _ in TIME_KEYWORDS]
    raise ValueError(
        f"No usable observation time in the header (tried {tried})."
        "Set DATE_OBS in main.py, e.g. DATE_OBS = '2018-12-30T01:58:54' "
        "(UTC), or add the right keyword to TIME_KEYWORDS in metadata.py."
    )


def get_exposure_time(header):
    """Return (exposure time in seconds, keyword) or (None, None)."""
    for key in EXPTIME_KEYWORDS:
        if key not in header:
            continue
        try:
            value = float(header[key])
        except (TypeError, ValueError):
            continue
        if _is_finite(value) and value > 0:
            return value * u.s, key
    return None, None


def _get_profile_exposure_time(header, profile):
    """Return a profile-specific total exposure value, when defined."""
    for key in profile.texptime_cards:
        if key not in header:
            continue
        try:
            value = float(header[key])
        except (TypeError, ValueError):
            continue
        if _is_finite(value) and value > 0:
            return value * u.s, key
    return None, None


def _select_trail_duration(
    profile,
    texptime,
    texptime_source,
    observation_duration,
    end_key,
    exptime,
):
    interval_source = (
        f"observation interval from {end_key}"
        if end_key
        else "observation interval from timestamp and exposure"
    )
    sources = {
        "texptime": (texptime, texptime_source),
        "observation_interval": (observation_duration, interval_source),
        "exptime": (exptime, "exposure-time card"),
    }
    preference_order = (
        ("texptime", "observation_interval", "exptime")
        if profile.trail_duration_preference == "texptime"
        else ("observation_interval", "texptime", "exptime")
    )
    for key in preference_order:
        duration, source = sources[key]
        if duration is not None:
            return duration, source
    return None, None


def get_obs_times(header, time_reference=None, date_obs_override=None):
    """Compatibility helper returning the historical five observation fields."""
    date_obs, start_obs, end_obs, exptime, source, _ = _get_observation_times(
        header, time_reference, date_obs_override
    )
    return date_obs, start_obs, end_obs, exptime, source


def _get_observation_times(header, time_reference, date_obs_override):
    if time_reference not in (None, *TIME_REFERENCES):
        raise ValueError(f"time_reference must be None or one of {list(TIME_REFERENCES)}")

    timesys = str(header.get("TIMESYS", "UTC")).strip().upper()
    if timesys != "UTC":
        print(f"WARNING: TIMESYS is {timesys!r}, but the time is treated as UTC.")

    date_obs, key, keyword_meaning, comment = get_header_time(
        header, date_obs_override
    )
    if time_reference:
        reference, why = time_reference, "set manually"
    elif keyword_meaning:
        reference, why = keyword_meaning, f"by definition of {key}"
    else:
        reference = _meaning_from_comment(comment)
        if reference:
            why = f"from the header comment {comment!r}"
        else:
            reference = "start"
            note = f"header comment: {comment!r}" if comment else "no header comment"
            why = f"ASSUMED, as in the FITS convention; {note}"

    origin = key if key == "set manually" else f"keyword {key}"
    source = f"{origin}, taken as the {reference} of the exposure ({why})"
    exptime, exp_key = get_exposure_time(header)
    if exptime is None:
        print(
            f"WARNING: no exposure time found (tried {list(EXPTIME_KEYWORDS)}). "
            "Start/end are taken from explicit header timestamps when available."
        )
        duration_source = source + "; no exposure time"
    else:
        duration_source = source + f"; exposure from {exp_key}"

    if reference == "start":
        start_obs = date_obs
        calculated_end = date_obs + exptime if exptime is not None else date_obs
    elif reference == "end":
        calculated_end = date_obs
        start_obs = date_obs - exptime if exptime is not None else date_obs
    else:
        start_obs = date_obs - exptime / 2 if exptime is not None else date_obs
        calculated_end = date_obs + exptime / 2 if exptime is not None else date_obs

    explicit_end, end_key = (
        (None, None)
        if key == "set manually"
        else _get_header_end_time(header)
    )
    end_obs = explicit_end if explicit_end is not None else calculated_end
    observation_duration = end_obs - start_obs
    if observation_duration.to_value(u.s) < 0:
        raise ValueError("The FITS observation end time precedes its start time.")
    if exptime is None and (end_key is None or reference != "start"):
        observation_duration = None
    if end_key:
        duration_source += f"; observation end from {end_key}"
    return (
        date_obs,
        start_obs,
        end_obs,
        exptime,
        duration_source,
        observation_duration,
    )


def _get_header_end_time(header):
    for key, kind in END_TIME_KEYWORDS:
        if key not in header:
            continue
        try:
            return _read_time(header[key], kind), key
        except (ValueError, TypeError):
            continue
    return None, None


def _read_time(value, kind):
    if kind == "jd":
        return Time(float(value), format="jd", scale="utc")
    if kind == "mjd":
        return Time(float(value), format="mjd", scale="utc")
    return Time(str(value).strip(), scale="utc")


def _comment(header, key):
    try:
        return str(header.comments[key]).strip()
    except (AttributeError, KeyError, IndexError):
        return ""


def _meaning_from_comment(comment):
    matches = [
        name
        for name, pattern in COMMENT_PATTERNS.items()
        if re.search(pattern, comment.lower())
    ]
    return matches[0] if len(matches) == 1 else None


def _get_observatory_code(header, profile, fallback_code, obscodes):
    for key in OBS_CODE_KEYWORDS:
        if key not in header:
            continue
        value = str(header[key]).strip()
        if OBS_CODE_PATTERN.match(value):
            return value.upper(), f"header keyword {key}"

    reason = "no list of observatory codes was given"
    if obscodes:
        normalized_header = {}
        for source, target in (
            ("TELESCOP", "TELESCOP"),
            ("OBSERVAT", "OBSERVAT"),
        ):
            if source in header:
                normalized_header[target] = header[source]
        for cards, target in (
            (profile.longitude_cards, "SITELON"),
            (profile.latitude_cards, "SITELAT"),
            (profile.altitude_cards, "OBSALT"),
        ):
            value = _first_number(header, cards)
            if value is not None:
                normalized_header[target] = value
        code, message = find_code(obscodes, normalized_header)
        if code:
            return code, message
        reason = message

    if fallback_code:
        return str(fallback_code).strip().upper(), "set manually"

    print(
        f"WARNING: no observatory code found ({reason}). "
        f"Using geocentric code {GEOCENTRIC_CODE}; "
        "use --observatory/-obs to set a different code."
    )
    return GEOCENTRIC_CODE, "geocentric fallback"


def _get_header_fwhm(header, profile):
    for key, unit in profile.fwhm_cards:
        if key not in header:
            continue
        try:
            value = float(header[key])
        except (TypeError, ValueError):
            continue
        if _is_finite(value) and value > 0:
            return value, unit, key
    return None, None, None


def _first_positive_number(header, keys):
    for key in keys:
        if key not in header:
            continue
        try:
            value = float(header[key])
        except (TypeError, ValueError):
            continue
        if _is_finite(value) and value > 0:
            return value
    return None


def _first_number(header, keys):
    for key in keys:
        if key not in header:
            continue
        try:
            value = float(header[key])
        except (TypeError, ValueError):
            continue
        if _is_finite(value):
            return value
    return None


def _is_finite(value):
    return math.isfinite(value)


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
    rate_sources = [source for source, clue in clues if _is_rate_unit_clue(clue)]
    if len(distinct_kinds) == 1:
        kind = next(iter(distinct_kinds))
        if kind in {"adu", "electron"} and rate_sources:
            sources = ", ".join(
                dict.fromkeys([source for _, source in candidates] + rate_sources)
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
        return (
            "adu",
            bunit,
            ", ".join(adu_evidence),
            "inferred as integrated ADU/DN from ADU-specific header comments",
        )

    if not clues:
        reason = "BUNIT and explicit unit-related comments are absent"
    elif any(re.search(r"\bcounts?\b", clue, re.IGNORECASE) for _, clue in clues):
        reason = "generic count units do not distinguish ADU/DN from electrons"
    else:
        reason = "unit metadata does not identify a supported image unit"
    return "unknown", bunit, None, reason


def _adu_header_evidence(header):
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


def _is_image_unit_description(text):
    unit = r"(?:\b(?:adus?|dns?|electrons?|counts?)\b|(?<![a-z])e-(?=\s|/|$))"
    description = r"\b(?:image|pixel|data)\b.{0,40}" + unit
    reverse = unit + r".{0,40}\b(?:image|pixel|data)\b"
    return bool(re.search(f"(?:{description}|{reverse})", text, re.IGNORECASE))


def _classify_unit_clue(clue):
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


def _is_rate_unit_clue(clue):
    normalized = clue.lower().replace("−", "-").replace("⁻¹", "^-1")
    return bool(
        re.search(
            r"(?:/|per\s*)(?:s|sec(?:ond)?s?)\b"
            r"|(?:^|[\s*])(?:s|sec(?:ond)?)\s*(?:\^|\*\*)?\s*-1(?:\b|$)",
            normalized,
        )
    )
