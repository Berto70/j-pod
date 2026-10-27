"""Queries to the JPL APIs (SB Identification + Horizons), and table preparation.

The SkyBoT (IMCCE) service is unavailable, so ``query_skybot()`` is now a
drop-in replacement based on the NASA/JPL Small-Body Identification API
(https://ssd-api.jpl.nasa.gov/sb_ident.api).  It returns a QTable with the
same column names and units the rest of the pipeline expects from SkyBoT:
Name, Number, RA, DEC, posunc, V, RA_rate, DEC_rate, sky_speed.
"""

import re
import requests
import time

import astropy.units as u
import numpy as np
from astroquery.jplhorizons import Horizons

from astropy.coordinates import SkyCoord, Angle
from astropy.table import QTable


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

# ---------------------------------------------------------------- JPL asteroid identification
#
# Drop-in replacement for the (currently unavailable) SkyBoT cone search.

JPL_SB_IDENT_URL = "https://ssd-api.jpl.nasa.gov/sb_ident.api"

SKYBOT_COLUMNS = (
    ("RA", u.deg),
    ("DEC", u.deg),
    ("posunc", u.arcsec),
    ("V", u.mag),
    ("RA_rate", u.arcsec / u.min),
    ("DEC_rate", u.arcsec / u.min),
    ("sky_speed", u.arcsec / u.min),
)


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
    """Format declination as [s]dd-mm-ss.ss for the JPL SB Identification API.

    The API wants the sign first and '-' as the digit separator; it accepts
    both '-dd-mm-ss.ss' and '+dd-mm-ss.ss' (verified against the live service).
    """
    return Angle(
        dec_deg, unit=u.deg
    ).to_string(
        unit=u.deg,
        sep="-",
        precision=2,
        alwayssign=True,
        pad=True,
    )


def _jpl_parse_dec(value):
    """Parse a JPL declination string into decimal degrees, or None.

    Accepted forms include "+10 28'34.3\"", "-16 41'45\"" and "M16-41-45";
    the leading sign (or the JPL 'M' = South marker) gives the hemisphere.
    """
    text = str(value).strip().replace(",", ".")  # e.g. '-16 41'45,'
    match = re.match(r"^([+-]|[MSms]?)(\d{1,2})(?:\D+(\d{1,2}))?(?:\D+([\d.]+))?\D*$", text)
    if match is None:
        return None

    sign, degrees, minutes, seconds = match.groups()
    try:
        result = float(degrees)
        if minutes:
            result += float(minutes) / 60.0
        if seconds:
            result += float(seconds) / 3600.0
    except ValueError:
        return None

    if sign in {"-", "M", "m"}:
        result = -result
    if not -90.0 <= result <= 90.0:
        return None
    return result


def _jpl_parse_ra(value):
    """Parse a JPL right-ascension string into hours, or None.

    Accepts separators between the components to be robust against format
    variations ('11:59:10', '11-59-10.00', '11 59 10.0').
    """
    text = str(value).strip().replace(",", ".")
    match = re.match(r"^(\d{1,2})\D+(\d{1,2})(?:\D+([\d.]+))?\D*$", text)
    if match is None:
        return None

    hours, minutes, seconds = match.groups()
    try:
        result = float(hours) + float(minutes) / 60.0
        if seconds:
            result += float(seconds) / 3600.0
    except ValueError:
        return None

    if not 0.0 <= result < 24.0:
        return None
    return result


def _to_float_or_nan(value):
    """Best-effort conversion of an API cell to a finite float, else NaN."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return np.nan
    return result if np.isfinite(result) else np.nan


def _empty_skybot_table():
    """An empty table with the SkyBoT column names and units."""
    table = QTable()
    table["Name"] = np.array([], dtype="U")
    table["Number"] = np.array([], dtype="U")
    for name, unit in SKYBOT_COLUMNS:
        table[name] = np.array([], dtype=float) * unit
    return table


def _fetch_sb_ident(params, timeout):
    """Query the JPL SB Identification API with retries; return the payload.

    HTTP 400/404 are not retried: they mean a malformed request (or an empty
    result), and the caller decides what to do with them.
    """
    session = requests.Session()
    session.headers.update({"User-Agent": "J-POD/1.0 (asteroid observation pipeline)"})
    max_attempts = 3

    for attempt in range(max_attempts):
        try:
            response = session.post(JPL_SB_IDENT_URL, data=params, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            if attempt == max_attempts - 1:
                raise RuntimeError(
                    f"JPL asteroid identification failed after {max_attempts} attempts."
                ) from exc
            time.sleep(2 ** (attempt + 1))
        except requests.exceptions.HTTPError as exc:
            # The API reports "no objects found" as HTTP 400 Code=7.
            if exc.response.status_code == 400 and '"Code":"7"' in exc.response.text:
                return {}
            retryable = exc.response.status_code in (429, 500, 502, 503, 504)
            if not retryable or attempt == max_attempts - 1:
                raise
            time.sleep(2 ** (attempt + 1))

    raise RuntimeError("unreachable state in the JPL SB_IDENT retry loop")


def query_skybot(field, radius, epoch, observatory_code, timeout=120):
    """Find Solar-system objects in a circular cone, mimicking SkyBot.cone_search.

    Because the IMCCE SkyBoT service is unavailable, this uses the NASA/JPL
    Small-Body Identification API instead: it queries the bounding box of the
    cone (two-pass mode, first pass suppressed) and then keeps only the
    objects whose true angular separation from ``field`` is within ``radius``.

    Parameters (same as the old SkyBoT version):
    - ``field``: scalar SkyCoord, the center of the cone.
    - ``radius``: angular radius of the cone (any angular Quantity).
    - ``epoch``: astropy Time of the observation.
    - ``observatory_code``: MPC observatory code, e.g. "I41".

    Returns a QTable with the SkyBoT column names and units:
    Name, Number (strings), RA, DEC (deg), posunc (arcsec), V (mag),
    RA_rate, DEC_rate, sky_speed (arcsec/min).  The motion columns are NaN:
    they are filled with consistent JPL Horizons values later, in
    ``fetch_jpl_data()``.  ``posunc`` is the quadrature sum of the API's
    estimated RA/Dec errors when available, otherwise NaN.
    """
    if not field.isscalar:
        raise ValueError("field must contain a single SkyCoord")

    radius = u.Quantity(radius).to(u.deg)
    radius_deg = radius.value
    if not 0 < radius_deg <= 180:
        raise ValueError("radius must be between 0 and 180 degrees")

    center = field.icrs
    ra_deg = float(center.ra.wrap_at(360 * u.deg).deg)
    dec_deg = float(center.dec.deg)

    # Bounding box of the cone.  The extra arcminutes on the RA half-width
    # compensate for the cos(dec) convergence at the edges of the box.
    cos_dec = max(abs(np.cos(np.deg2rad(dec_deg))), 1e-6)
    ra_half_width = min(radius_deg / cos_dec + 2.0 * (1.0 / 60.0), 180.0)

    epoch_string = epoch.utc.to_datetime().strftime("%Y-%m-%d_%H:%M:%S")

    params = {
        "sb-kind": "a",   # SB_IDENT only accepts 'a' or 'c'; asteroids match
                          # the main use of the old SkyBoT cone search here,
                          # and comets are still resolved via JPL Horizons in
                          # fetch_jpl_data() when an object is cross-matched.
        "mpc-code": str(observatory_code),
        "obs-time": epoch_string,
        "fov-ra-center": _jpl_ra_string(ra_deg),
        "fov-dec-center": _jpl_dec_string(dec_deg),
        "fov-ra-hwidth": f"{ra_half_width:.4f}",
        "fov-dec-hwidth": f"{radius_deg:.4f}",
        "two-pass": "true",
        "suppress-first-pass": "true",
        "req-elem": "false",
        "mag-required": "false",
    }

    payload = _fetch_sb_ident(params, timeout)

    if payload.get("error") or payload.get("code"):
        message = payload.get("message") or payload.get("error") or payload.get("code")
        raise RuntimeError(f"JPL asteroid identification failed: {message}")

    signature = payload.get("signature", {})
    source = signature.get("source", "") if isinstance(signature, dict) else str(signature)
    if "Small-Body Identification" not in source:
        raise RuntimeError(f"Unexpected JPL API response signature: {signature!r}")

    fields = payload.get("fields_second") or payload.get("fields_first") or []
    rows = payload.get("data_second_pass")
    if rows is None:
        rows = payload.get("data_first_pass", [])

    if not rows:
        print("JPL SB_IDENT found no objects in this field.")
        return _empty_skybot_table()

    index = {name: i for i, name in enumerate(fields)}

    def find_column(*fragments):
        """Return the index of the first field containing all fragments."""
        for name, position in index.items():
            lowered = name.lower()
            if all(fragment.lower() in lowered for fragment in fragments):
                return position
        return None

    col_name = find_column("object name")
    col_ra = find_column("astrometric ra")
    col_dec = find_column("astrometric dec")
    if col_name is None or col_ra is None or col_dec is None:
        raise RuntimeError(f"Unexpected JPL SB_IDENT columns: {fields}")

    col_mag = find_column("visual magnitude")
    col_err_ra = find_column("est. error ra")
    col_err_dec = find_column("est. error dec")

    names, numbers, ra_values, dec_values, mag_values, posunc_values = [], [], [], [], [], []
    for row in rows:
        name = str(row[col_name]).strip()
        ra_hours = _jpl_parse_ra(row[col_ra])
        dec_deg_row = _jpl_parse_dec(row[col_dec])
        if not name or ra_hours is None or dec_deg_row is None:
            continue

        magnitude = _to_float_or_nan(row[col_mag]) if col_mag is not None else np.nan
        if col_err_ra is not None and col_err_dec is not None:
            err_ra = _to_float_or_nan(row[col_err_ra])
            err_dec = _to_float_or_nan(row[col_err_dec])
            posunc = np.hypot(err_ra, err_dec)
        else:
            posunc = np.nan

        names.append(name)
        # A leading number outside parentheses is the (provisional-)number part
        # of the designation, e.g. "1301 Yvonne (1934 EA)" -> 1301; parenthesized
        # designations like "(2020 KP17)" have no catalogue number.
        match = re.match(r"^(\d+)\s", name)
        numbers.append(match.group(1) if match else "")
        ra_values.append(ra_hours)
        dec_values.append(dec_deg_row)
        mag_values.append(magnitude)
        posunc_values.append(posunc)

    if not names:
        print("JPL SB_IDENT returned no usable positions in this field.")
        return _empty_skybot_table()

    coordinates = SkyCoord(ra=ra_values * u.hourangle,
                           dec=dec_values * u.deg,
                           frame="icrs")

    # The API searches a rectangular box; reduce the result to the true cone.
    inside = center.separation(coordinates) <= radius
    coordinates = coordinates[inside]
    names = [n for n, k in zip(names, inside) if k]
    numbers = [n for n, k in zip(numbers, inside) if k]
    mag_values = [m for m, k in zip(mag_values, inside) if k]
    posunc_values = [p for p, k in zip(posunc_values, inside) if k]

    table = QTable()
    table["Name"] = names
    table["Number"] = numbers
    table["RA"] = coordinates.ra.to(u.deg)
    table["DEC"] = coordinates.dec.to(u.deg)
    table["posunc"] = np.asarray(posunc_values, dtype=float) * u.arcsec
    table["V"] = np.asarray(mag_values, dtype=float) * u.mag

    # Motion columns start as NaNs; they are overwritten by JPL Horizons data
    # in fetch_jpl_data() to ensure data consistency across the pipeline.
    n = len(table)
    table["RA_rate"] = np.full(n, np.nan) * (u.arcsec / u.min)
    table["DEC_rate"] = np.full(n, np.nan) * (u.arcsec / u.min)
    table["sky_speed"] = np.full(n, np.nan) * (u.arcsec / u.min)

    print(f"JPL SB_IDENT returned {n} asteroids inside a "
          f"{u.Quantity(radius).to(u.arcmin):.2f} cone.")
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
    """Add H, SMAA_3sigma, SMIA_3sigma and LOV_PA columns, filled from JPL.

    Also refreshes RA, DEC, V, RA_rate, DEC_rate and sky_speed with the
    Horizons ephemeris of each object, so that all astrometry in the table
    is consistent with the uncertainty ellipse.  Every column keeps its
    unit; values are always written as Quantity objects (or through the
    raw ndarray view), never as bare floats into a unit-ed column.
    """
    if not isinstance(table, QTable):
        table = QTable(table)

    n = len(table)

    # Guarantee units on every column this function touches, so later
    # assignments and downstream code (plotting, detection, CSV) see
    # proper Quantities instead of dimensionless arrays.
    for name, unit in SKYBOT_COLUMNS:
        if name not in table.colnames:
            table[name] = np.full(n, np.nan) * unit
        elif table[name].unit is None:
            # A unit-less column (e.g. rebuilt from a CSV without units) is
            # interpreted as being already expressed in the expected unit.
            table[name] = table[name].data.view(np.ndarray).astype(float) * unit
        else:
            table[name] = table[name].to(unit)

    # Initialize the JPL columns with explicit units.
    for name, unit in JPL_COLUMNS:
        table[name] = np.full(n, np.nan) * unit

    print(f"\nFound {n} objects. Fetching H, astrometry and uncertainty ellipses from JPL Horizons...")

    cache = {}
    for i in range(n):
        catalog_name = str(table["Name"][i]).strip()
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
            # query_jpl_horizons returns plain floats already expressed in the
            # units below; attach the unit explicitly when writing into the
            # (unit-ed) columns, which is what Quantity.__setitem__ requires.
            for name, unit in JPL_COLUMNS:
                value = jpl_data[name]
                if value is not None:
                    table[name][i] = float(value) * unit

            # JPL astrometry and apparent magnitude (consistency with the ellipse)
            if jpl_data.get("RA_JPL") is not None:
                table["RA"][i] = float(jpl_data["RA_JPL"]) * u.deg
            if jpl_data.get("DEC_JPL") is not None:
                table["DEC"][i] = float(jpl_data["DEC_JPL"]) * u.deg
            if jpl_data.get("V_JPL") is not None:
                table["V"][i] = float(jpl_data["V_JPL"]) * u.mag

            # Sky motion: JPL returns arcsec/hour, we store arcsec/min
            ra_rate = dec_rate = None
            if jpl_data.get("RA_rate_JPL") is not None:
                ra_rate = (float(jpl_data["RA_rate_JPL"]) * (u.arcsec / u.hour)).to_value(u.arcsec / u.min)
                table["RA_rate"][i] = ra_rate * (u.arcsec / u.min)
            if jpl_data.get("DEC_rate_JPL") is not None:
                dec_rate = (float(jpl_data["DEC_rate_JPL"]) * (u.arcsec / u.hour)).to_value(u.arcsec / u.min)
                table["DEC_rate"][i] = dec_rate * (u.arcsec / u.min)
            if ra_rate is not None and dec_rate is not None:
                speed = np.hypot(ra_rate, dec_rate)
                table["sky_speed"][i] = speed * (u.arcsec / u.min)

            print(f" [{i + 1}/{n}] {catalog_name} "
                  f"(JPL identifier {matched_identifier}): {_describe(jpl_data)}")
        elif last_error is not None:
            print(f" [{i + 1}/{n}] {catalog_name}: JPL query failed ({last_error})")
        else:
            print(f" [{i + 1}/{n}] {catalog_name}: "
                  "no usable identifier was available for the JPL query")

    return table
