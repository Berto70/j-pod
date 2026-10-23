"""Queries to SkyBoT (IMCCE) and JPL Horizons, and table preparation."""

import re

import astropy.units as u
import numpy as np
from astroquery.imcce import Skybot
from astroquery.jplhorizons import Horizons


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

def query_skybot(field, radius, epoch, observatory_code):
    """Cone search in SkyBoT. If anything is found, standardize the table
    (motion columns and 'posunc' in arcsec)."""
    table = Skybot.cone_search(field, radius, epoch, location=observatory_code)
    if len(table) > 0:
        table["posunc"] = u.Quantity(table["posunc"]).to(u.arcsec)

        # initialize motion columns with NaNs. 
        # they will be overwritten by JPL data in fetch_jpl_data
        # to ensure data consistency
        n = len(table)
        table["RA_rate"] = np.full(n, np.nan) * (u.arcsec / u.min)
        table["DEC_rate"] = np.full(n, np.nan) * (u.arcsec / u.min)
        table["sky_speed"] = np.full(n, np.nan) * (u.arcsec / u.min)

    return table


# ---------------------------------------------------------------- JPL

def query_jpl_horizons(identifier, epoch, location):
    """Fetch H and the plane-of-sky 3-sigma uncertainty ellipse from JPL."""
    ephemeris = Horizons(
        id=identifier,
        id_type="smallbody",
        location=location,
        epochs=float(epoch.jd),
    ).ephemerides()

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
    n = len(table)
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
            # uncertainty and H
            for name, unit in JPL_COLUMNS:
                if jpl_data[name] is not None:
                    table[name][i] = jpl_data[name] * unit

            # jpl astrometry and apparent mag
            if jpl_data.get("RA_JPL") is not None:
                table["RA"][i] = jpl_data["RA_JPL"] * u.deg
            if jpl_data.get("DEC_JPL") is not None:
                table["DEC"][i] = jpl_data["DEC_JPL"] * u.deg
            if jpl_data.get("V_JPL") is not None:
                table["V"][i] = jpl_data["V_JPL"] if table["V"].unit is None else jpl_data["V_JPL"] * u.mag

            # sky motion
            # JPL returns arcsec/hour, but we want arcsec/min
            if jpl_data.get("RA_rate_JPL") is not None:
                table["RA_rate"][i] = (jpl_data["RA_rate_JPL"] * (u.arcsec / u.hour)).to(u.arcsec / u.min)
            if jpl_data.get("DEC_rate_JPL") is not None:
                table["DEC_rate"][i] = (jpl_data["DEC_rate_JPL"] * (u.arcsec / u.hour)).to(u.arcsec / u.min) 

            # calc sky_speed
            ra_r = u.Quantity(table["RA_rate"][i]).value
            dec_r = u.Quantity(table["DEC_rate"][i]).value
            table["sky_speed"][i] = np.sqrt(ra_r**2 + dec_r**2) * (u.arcsec / u.min)
            
            print(f" [{i + 1}/{n}] {skybot_name} "
                  f"(JPL identifier {matched_identifier}): {_describe(jpl_data)}")
        elif last_error is not None:
            print(f" [{i + 1}/{n}] {skybot_name}: JPL query failed ({last_error})")
        else:
            print(f" [{i + 1}/{n}] {skybot_name}: "
                  "no usable identifier was available for the JPL query")
