"""Queries to SkyBoT (IMCCE) and JPL Horizons, and table preparation."""

import re
import requests
import time

import astropy.units as u
import numpy as np
from astroquery.imcce import Skybot
from astroquery.jplhorizons import Horizons

from astropy.coordinates import SkyCoord, Angle
from astropy.table import Table, QTable


# ---------------------------------------------------------------- helpers

def _usable_text(value):
    """Return a cleaned catalogue value, or None for a missing value."""
    if value is None or np.ma.is_masked(value):
        return None

    text = str(value).strip()
    if text.lower() in {"", "-", "--", "nan", "none"}:
        return None
    return text


def jpl_identifiers(table, row):
    """Build JPL query identifiers, preferring SkyBoT's asteroid number."""
    identifiers = []

    if "Number" in table.colnames:
        number = _usable_text(table["Number"][row])
        if number:
            # Some table readers represent an integer catalogue number as 1.0.
            if re.fullmatch(r"\d+\.0+", number):
                number = number.split(".", 1)[0]
            if number.isdigit():
                identifiers.append(number)

    name = _usable_text(table["Name"][row])
    if name:
        # Extract a number only when it is explicitly parenthesized, as in
        # "(433) Eros".  A provisional designation such as "2006 BU4" must
        # remain intact: its leading year is not an asteroid number.
        match = re.match(r"^\(\s*(\d+)\s*\)", name)
        if match:
            identifiers.append(match.group(1))
        identifiers.append(name)

    # Retain order while removing duplicates.
    return list(dict.fromkeys(identifiers))


def _table_value(table, column, unit=None):
    """Return one finite scalar from a one-row table, or None."""
    if column not in table.colnames or len(table) == 0:
        return None

    column_data = table[column]
    value = column_data[0]
    if value is None or np.ma.is_masked(value):
        return None

    try:
        if unit is not None and hasattr(value, "to_value"):
            result = float(value.to_value(unit))
        elif unit is not None and getattr(column_data, "unit", None) is not None:
            result = float((value * column_data.unit).to_value(unit))
        else:
            result = float(value)
    except (TypeError, ValueError, u.UnitConversionError):
        return None

    return result if np.isfinite(result) else None

# ---------------------------------------------------------------- SkyBoT

# def query_skybot(field, radius, epoch, observatory_code):
#     """Cone search in SkyBoT. If anything is found, standardize the table
#     (motion columns and 'posunc' in arcsec)."""
#     table = Skybot.cone_search(field, radius, epoch, location=observatory_code)
#     if len(table) > 0:
#         table["posunc"] = u.Quantity(table["posunc"]).to(u.arcsec)

#         # initialize motion columns with NaNs. 
#         # they will be overwritten by JPL data in fetch_jpl_data
#         # to ensure data consistency
#         n = len(table)
#         table["RA_rate"] = np.full(n, np.nan) * (u.arcsec / u.min)
#         table["DEC_rate"] = np.full(n, np.nan) * (u.arcsec / u.min)
#         table["sky_speed"] = np.full(n, np.nan) * (u.arcsec / u.min)

#     return table

# ---------------------------------------------------------------- JPL asteroid identification

JPL_SB_IDENT_URL = "https://ssd-api.jpl.nasa.gov/sb_ident.api"


def _jpl_ra_string(ra_deg):
    """Format RA as hh-mm-ss.ss for the JPL SB Identification API."""
    return Angle(
        ra_deg, unit=u.deg
    ).wrap_at(360 * u.deg).to_string(
        unit=u.hourangle,
        sep="-",
        precision=2,
        pad=True,
    )


def _jpl_dec_string(dec_deg):
    """Format declination as dd-mm-ss.ss, using M for negative values."""
    value = Angle(
        dec_deg, unit=u.deg
    ).to_string(
        unit=u.deg,
        sep="-",
        precision=2,
        alwayssign=True,
        pad=True,
    )

    if value.startswith("-"):
        value = "M" + value[1:]

    return value


def _jpl_dec_to_colon(value):
    """Convert a JPL declination such as +10 28'34.3" to +10:28:34.3."""
    return (
        str(value)
        .replace("'", ":")
        .replace('"', "")
        .replace(" ", ":")
    )


def query_skybot(
    field,
    radius,
    epoch,
    observatory_code,
    timeout=120,
):
    """
    Find asteroids in a circular cone using the JPL SB Identification API.
    Returns a QTable with strict unit enforcement to match J-POD's expectations.
    """
    if not field.isscalar:
        raise ValueError("field must contain a single SkyCoord")

    radius_deg = u.Quantity(radius).to_value(u.deg)
    if not 0 < radius_deg <= 180:
        raise ValueError("radius must be between 0 and 180 degrees")

    ra_deg = field.icrs.ra.deg
    dec_deg = field.icrs.dec.deg

    cos_dec = abs(np.cos(np.deg2rad(dec_deg)))
    ra_half_width = 180.0 if cos_dec < 1e-10 else min(radius_deg / cos_dec, 180.0)
    epoch_string = epoch.utc.to_datetime().strftime("%Y-%m-%d_%H:%M:%S")

    params = {
        "sb-kind": "a",
        "mpc-code": str(observatory_code),
        "obs-time": epoch_string,
        "fov-ra-center": _jpl_ra_string(ra_deg),
        "fov-dec-center": _jpl_dec_string(dec_deg),
        "fov-ra-hwidth": ra_half_width,
        "fov-dec-hwidth": radius_deg,
        "two-pass": "true",
        "suppress-first-pass": "true",
        "req-elem": "false",
        "mag-required": "false",
    }

    session = requests.Session()
    session.headers.update({"User-Agent": "J-POD/1.0 (asteroid observation pipeline)"})
    max_attempts = 3

    for attempt in range(max_attempts):
        try:
            response = session.get(JPL_SB_IDENT_URL, params=params, timeout=(15, 300))
            response.raise_for_status()
            payload = response.json()
            break
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            if attempt == max_attempts - 1:
                raise RuntimeError(f"JPL asteroid identification failed after {max_attempts} attempts.") from exc
            time.sleep(2 ** (attempt + 1))
        except requests.exceptions.HTTPError as exc:
            if response.status_code not in (429, 500, 502, 503, 504) or attempt == max_attempts - 1:
                raise
            time.sleep(2 ** (attempt + 1))
    else:
        raise RuntimeError("JPL SB_IDENT failed without returning a response")

    if payload.get("code") or payload.get("error"):
        raise RuntimeError(f"JPL asteroid search failed: {payload.get('message', payload.get('error', payload.get('code')))}")

    signature = payload.get("signature", {})
    if signature.get("source") != "NASA/JPL Small-Body Identification API":
        raise RuntimeError(f"Unexpected JPL API response: {signature}")

    rows = payload.get("data_second_pass", [])
    columns = payload.get("fields_second", [])

    # CRITICAL FIX: Return a QTable with explicit units, even if empty
    if not rows:
        table = QTable()
        table["Name"] = np.array([], dtype=str)
        table["Number"] = np.array([], dtype=str)
        table["RA"] = np.array([], dtype=float) * u.deg
        table["DEC"] = np.array([], dtype=float) * u.deg
        table["posunc"] = np.array([], dtype=float) * u.arcsec
        table["V"] = np.array([], dtype=float) * u.mag
        table["RA_rate"] = np.array([], dtype=float) * (u.arcsec / u.min)
        table["DEC_rate"] = np.array([], dtype=float) * (u.arcsec / u.min)
        table["sky_speed"] = np.array([], dtype=float) * (u.arcsec / u.min)
        return table

    raw = QTable(rows=rows, names=columns)
    ra_col, dec_col, name_col, mag_col = "Astrometric RA (hh:mm:ss)", 'Astrometric Dec (dd mm\'ss")', "Object name", "Visual magnitude (V)"

    for required in (ra_col, dec_col, name_col):
        if required not in raw.colnames:
            raise RuntimeError(f"Expected column {required!r} missing from JPL response.")

    ra_strings = [str(value) for value in raw[ra_col]]
    dec_strings = [_jpl_dec_to_colon(value) for value in raw[dec_col]]
    coordinates = SkyCoord(ra=ra_strings, dec=dec_strings, unit=(u.hourangle, u.deg), frame="icrs")

    names = [str(value).strip() for value in raw[name_col]]
    numbers = [re.match(r"^\s*(\d+)(?:\s|$)", name).group(1) if re.match(r"^\s*(\d+)(?:\s|$)", name) else "" for name in names]

    # CRITICAL FIX: Use QTable and explicitly attach units to every column
    table = QTable()
    table["Name"] = names
    table["Number"] = numbers
    table["RA"] = coordinates.ra.deg * u.deg       # Explicitly degrees
    table["DEC"] = coordinates.dec.deg * u.deg     # Explicitly degrees
    table["posunc"] = np.full(len(table), np.nan) * u.arcsec

    if mag_col in raw.colnames:
        magnitudes = []
        for value in raw[mag_col]:
            try:
                mag = float(value)
                magnitudes.append(mag if np.isfinite(mag) else np.nan)
            except (TypeError, ValueError):
                magnitudes.append(np.nan)
        table["V"] = np.asarray(magnitudes, dtype=float) * u.mag
    else:
        table["V"] = np.full(len(table), np.nan) * u.mag

    separation = field.icrs.separation(coordinates)
    table = table[separation <= radius]

    n = len(table)
    table["RA_rate"] = np.full(n, np.nan) * (u.arcsec / u.min)
    table["DEC_rate"] = np.full(n, np.nan) * (u.arcsec / u.min)
    table["sky_speed"] = np.full(n, np.nan) * (u.arcsec / u.min)

    print(f"JPL SB_IDENT returned {len(table)} asteroids inside a {radius.to(u.arcmin):.2f} cone.")
    return table


# ---------------------------------------------------------------- JPL

def query_jpl_horizons(identifier, epoch, location):
    """Fetch H and the plane-of-sky 3-sigma uncertainty ellipse from JPL."""
    ephemeris = Horizons(
        id=identifier,
        id_type="smallbody",
        location=location,
        epochs=float(epoch.jd),
    ).ephemerides(quantities="1,3,9,36,37,47") # query only quantities we need

    h_value = _table_value(ephemeris, "H", u.mag)
    major = _table_value(ephemeris, "SMAA_3sigma", u.arcsec)
    minor = _table_value(ephemeris, "SMIA_3sigma", u.arcsec)
    theta = _table_value(ephemeris, "Theta_3sigma", u.deg)

    # Horizons Theta is measured from increasing RA toward increasing DEC.
    # Convert it to the usual astronomical position angle: east of north.
    lov_pa = None if theta is None else (90.0 - theta) % 180.0
    
    # fetch astrometry and apparent mag from JPL
    # ensures consistency with the uncertainty ellipse
    jpl_ra = _table_value(ephemeris, "RA", u.deg)
    jpl_dec = _table_value(ephemeris, "DEC", u.deg)

    # fetch sky motion from JPL
    jpl_ra_rate = _table_value(ephemeris, "RA_rate", u.arcsec / u.hour)
    jpl_dec_rate = _table_value(ephemeris, "DEC_rate", u.arcsec / u.hour)

    # check whether it's V or mag --> remove if there's no ambiguity
    jpl_v = _table_value(ephemeris, "V", u.mag)
    if jpl_v is None:
        jpl_v = _table_value(ephemeris, "mag", u.mag)

    return {
        "H": h_value,
        "SMAA_3sigma": major,
        "SMIA_3sigma": minor,
        "LOV_PA": lov_pa,
        "RA_JPL": jpl_ra,
        "DEC_JPL": jpl_dec,
        "V_JPL": jpl_v,
        "RA_rate_JPL": jpl_ra_rate,
        "DEC_rate_JPL": jpl_dec_rate,
    }


JPL_COLUMNS = (
    ("H", u.mag),
    ("SMAA_3sigma", u.arcsec),
    ("SMIA_3sigma", u.arcsec),
    ("LOV_PA", u.deg),
)


def _describe(jpl_data):
    """Short text summarizing what JPL returned, for the progress log."""
    h_text = (f"{jpl_data['H']:.3f}" if jpl_data["H"] is not None
              else "not available")
    if all(jpl_data[k] is not None for k in ("SMAA_3sigma", "SMIA_3sigma", "LOV_PA")):
        ellipse_text = (
            f"3-sigma ellipse = {jpl_data['SMAA_3sigma']:.3f} x "
            f"{jpl_data['SMIA_3sigma']:.3f} arcsec, "
            f"LOV PA = {jpl_data['LOV_PA']:.3f} deg"
        )
    else:
        ellipse_text = "uncertainty ellipse not available"
    return f"H = {h_text}; {ellipse_text}"

def fetch_jpl_data(table, epoch, observatory_code):
    """Add H, SMAA_3sigma, SMIA_3sigma and LOV_PA columns, filled from JPL."""
    # 1. GUARANTEE QTable to prevent Astropy from stripping units
    if not isinstance(table, QTable):
        table = QTable(table)

    n = len(table)
    
    # 2. Enforce units on core columns (patching any dimensionless leftovers)
    for name in ("RA", "DEC"):
        if table[name].unit is None:
            table[name] = table[name] * u.deg
        else:
            table[name] = table[name].to(u.deg)
            
    if "V" in table.colnames and table["V"].unit is None:
        table["V"] = table["V"] * u.mag
    elif "V" not in table.colnames:
        table["V"] = np.full(n, np.nan) * u.mag

    for name in ("RA_rate", "DEC_rate", "sky_speed"):
        if name in table.colnames and table[name].unit is None:
            table[name] = table[name] * (u.arcsec / u.min)

    # 3. Initialize JPL columns with explicit units
    for name, unit in JPL_COLUMNS:
        table[name] = np.full(n, np.nan) * unit

    print(f"\nFound {n} objects. Fetching H, astrometry and uncertainty ellipses from JPL Horizons...")

    cache = {}
    for i in range(n):
        skybot_name = str(table["Name"][i]).strip()
        matched_identifier = None
        jpl_data = None
        last_error = None

        for identifier in jpl_identifiers(table, i):
            try:
                if identifier not in cache:
                    cache[identifier] = query_jpl_horizons(identifier, epoch, observatory_code)
                jpl_data = cache[identifier]
                matched_identifier = identifier
                break
            except Exception as exc:
                last_error = exc

        if jpl_data is not None:
            # 4. SAFE ASSIGNMENT: Assign raw float values, NOT Quantities
            for name, unit in JPL_COLUMNS:
                value = jpl_data[name]
                if value is not None:
                    if isinstance(value, u.Quantity):
                        value = value.to_value(unit)
                    table[name][i] = value  # Astropy safely maps the float to the column's unit

            # JPL astrometry and apparent mag (assign raw floats)
            if jpl_data.get("RA_JPL") is not None:
                table["RA"][i] = float(jpl_data["RA_JPL"])
            if jpl_data.get("DEC_JPL") is not None:
                table["DEC"][i] = float(jpl_data["DEC_JPL"])
            if jpl_data.get("V_JPL") is not None:
                # CRITICAL FIX: Assign the raw float, NOT the Quantity
                table["V"][i] = float(jpl_data["V_JPL"])

            # Sky motion: JPL returns arcsec/hour, we want arcsec/min
            if jpl_data.get("RA_rate_JPL") is not None:
                val = (float(jpl_data["RA_rate_JPL"]) * (u.arcsec / u.hour)).to_value(u.arcsec / u.min)
                table["RA_rate"][i] = val
            if jpl_data.get("DEC_rate_JPL") is not None:
                val = (float(jpl_data["DEC_rate_JPL"]) * (u.arcsec / u.hour)).to_value(u.arcsec / u.min)
                table["DEC_rate"][i] = val

            # Calc sky_speed
            ra_r = float(table["RA_rate"][i])
            dec_r = float(table["DEC_rate"][i])
            table["sky_speed"][i] = np.sqrt(ra_r**2 + dec_r**2)

            print(f" [{i + 1}/{n}] {skybot_name} "
                  f"(JPL identifier {matched_identifier}): {_describe(jpl_data)}")
        elif last_error is not None:
            print(f" [{i + 1}/{n}] {skybot_name}: JPL query failed ({last_error})")
        else:
            print(f" [{i + 1}/{n}] {skybot_name}: "
                  "no usable identifier was available for the JPL query")
