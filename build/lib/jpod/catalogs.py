"""Queries to SkyBoT (IMCCE) and JPL Horizons, and table preparation.
SkyBot is only used to find which objects are in the field. Everything else comes from JPL Horizons:
the positions at the start and at the end of the exposure, the rates, H and the uncertainty ellipse. 
In this way the ellipse is centered on the position of the same orbit solution.
If Horizons has nothing for an object, its SkyBot position and rate are used instead 
(the end position is the start position plus the rate times the exposure). The column 'source' tells
which one was used for each object.

"""

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
    """Build JPL query identifiers, preferring the catalog's asteroid number."""
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
        # JPL may wrap a provisional designation in parentheses, e.g.
        # "(2025 QC160)"; Horizons expects the designation without them.
        wrapped_designation = re.fullmatch(
            r"\(\s*(\d{4}\s+[A-Z]{1,2}\d+[A-Z]?)\s*\)", name
        )
        if wrapped_designation:
            name = wrapped_designation.group(1)

        # Extract a number only when it is explicitly parenthesized, as in
        # "(433) Eros".  A provisional designation such as "2006 BU4" must
        # remain intact: its leading year is not an asteroid number.
        match = re.match(r"^\(\s*(\d+)\s*\)", name)
        if match:
            identifiers.append(match.group(1))
        identifiers.append(name)

    # Retain order while removing duplicates.
    return list(dict.fromkeys(identifiers))


def _table_value(table, column, unit=None, row=0):
    """Return one finite scalar of the given row from a table or None."""
    if column not in table.colnames or len(table) == 0 or row >= len(table):
        return None

    column_data = table[column]
    value = column_data[row]
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

def _rate_arcsec_per_min(table, column, row=0):
    """Rate of a Horizons table in arcsec/min. Horizons gives arcsec/hour."""
    if column not in table.colnames or row>=len(table):
        return None
    column_data = table[column]
    value = column_data[row]
    if value is None or np.ma.is_masked(value):
        return None
    unit = getattr(column_data, "unit", None)
    if unit is None:
        unit = u.arcsec/u.hour
    try:
        quantity = value if hasattr(value, "to_value") else value * unit
        result = float(quantity.to_value(u.arcsec/u.min))
    except (TypeError, ValueError, u.UnitConversionError):
        return None
    return result if np.isfinite(result) else None

def add_sky_motion(table):
    """Add standardized motion components and total angular speed."""
    # astroquery versions use either RA_rate or RAcosD_rate for the first
    # component. SkyBoT's value already includes cos(DEC), so the two
    # components can be combined directly on the tangent plane.
    if "RA_rate" in table.colnames:
        ra_rate_name = "RA_rate"
    elif "RAcosD_rate" in table.colnames:
        ra_rate_name = "RAcosD_rate"
    else:
        raise KeyError("SkyBoT result has no RA motion-rate column")

    if "DEC_rate" not in table.colnames:
        raise KeyError("SkyBoT result has no DEC motion-rate column")

    ra_rate = u.Quantity(table[ra_rate_name]).to(u.arcsec / u.min)
    dec_rate = u.Quantity(table["DEC_rate"]).to(u.arcsec / u.min)

    table["RA_rate"] = ra_rate
    table["DEC_rate"] = dec_rate
    table["sky_speed"] = np.sqrt(ra_rate**2 + dec_rate**2).to(u.arcsec / u.min)

def _end_from_rates(ra_deg, dec_deg, ra_rate, dec_rate, exposure_s):
    """Position after 'exposure_s' seconds for the rates 'ra_rate' (which includes cos(dec)) and 'dec_rate', in arcsec/min.
    Returns (ra,dec) in degrees."""
    minutes = exposure_s/60.0
    d_east = np.asarray(ra_rate)*minutes/3600.0 # degrees
    d_north = np.asarray(dec_rate)*minutes/3600.0
    ra_end = np.asarray(ra_deg) +d_east/np.cos(np.radians(dec_deg))
    dec_end = np.asarray(dec_deg)+d_north
    return ra_end, dec_end

def _split_asteroid_type(table):
    """
    Split the 'Type' column into 'class' and 'sub-class'.

    Examples:
        NEA>Amor          -> NEA, Amor
        MB>Inner          -> MB, Inner
        KBO>Resonant>5:3  -> KBO, Resonant>5:3
        Mars-Crosser      -> MC, ''
        Trojan            -> Trojan, ''
        Hungaria          -> MB, Hungaria
    """
    standalone_classes = {
        "Trojan": ("Trojan",""),
        "Mars-Crosser": ("MC",""),
        "Hungaria": ("MB","Hungaria"),
        "Comet": ("Comet", ""),
    }

    classes = []
    subclasses = []
    unknown_types = set()

    for value in table["Type"]:
        if value is None or str(value).strip() == "":
            classes.append("")
            subclasses.append("")
            continue
        type = str(value).strip()
        if ">" in type:
            main_class, subclass = type.split(">",1)
            classes.append(main_class)
            subclasses.append(subclass)
        elif type in standalone_classes:
            main_class, subclass = standalone_classes[type]
            classes.append(main_class)
            subclasses.append(subclass)
        else:
            # Preserve unknown values rather than guessing their class
            classes.append(type)
            subclasses.append("")
            unknown_types.add(type)

    table["class"] = classes
    table["sub-class"] = subclasses
    if unknown_types:
        print("\nUnmapped standalone dynamical classes:")
        for value in sorted(unknown_types):
            print(f" - {value}")
        print()
    return table


# ---------------------------------------------------------------- SkyBoT

def query_skybot(field, radius, epoch, observatory_code):
    """Cone search in SkyBoT. If anything is found, standardize the table
    (motion columns and 'posunc' in arcsec)."""
    table = Skybot.cone_search(field, radius, epoch, location=observatory_code)
    if len(table) > 0:
        add_sky_motion(table)
        table["RA"] = u.Quantity(table["RA"]).to(u.deg)
        table["DEC"] = u.Quantity(table["DEC"]).to(u.deg)
        table["posunc"] = u.Quantity(table["posunc"]).to(u.arcsec)
        # Add the "Type" column that filters objects based on dynamical properties (MBA, NEA, ...)
        if "Type" in table.colnames:
            table = _split_asteroid_type(table)
    return table

def query_skybot_start_end(field, radius, start_obs, end_obs, observatory_code):
    """Cone search in SkyBot at the start and at the end of the exposure.
    Returns the table of the start of the exposure with two more columns, RA_end and DEC_end.
    Objects found at only one of the two times are dropped. They lie at the border of the cone, 
    so they are further from the image than the margin used for the search (as long as the margin is longer
    than the distance covered during the exposure) and cannot be inside it at both times.
    If start_obs and end_obs coincide (exposure time unknown) a single query is made and RA_end, DEC_end are copies of RA, DEC."""
    start_table = query_skybot(field, radius, start_obs, observatory_code)
    if len(start_table) == 0:
        return start_table
    if abs((end_obs-start_obs).sec) < 1e-6:
        start_table["RA_end"] = start_table["RA"]
        start_table["DEC_end"] = start_table["DEC"]
        return start_table
    end_table = query_skybot(field, radius, end_obs, observatory_code)
    if len(end_table) == 0:
        print("SkyBot: no objects at the end of the exposure")
        return start_table[:0]
    # Match the objects of the two tables by name
    end_index = {str(name).strip(): i for i, name in enumerate(end_table["Name"])}
    index = np.array([end_index.get(str(name).strip(),-1)
                      for name in start_table["Name"]])
    found = index >= 0
    table = start_table[found]
    table["RA_end"] = end_table["RA"][index[found]]
    table["DEC_end"] = end_table["DEC"][index[found]]
    # Add the "Type" column that filters objects based on dynamical properties (MBA, NEA, ...)
    if "Type" not in table.colnames and "Type" in start_table.colnames:
        table["Type"] = start_table["Type"][found]
    table = _split_asteroid_type(table)
    print(f"SkyBot: {len(start_table)} objects at the start of the exposure,"
          f"{len(end_table)} at the end; {len(table)} found at both times.")
    return table

# ---------------------------------------------------------------- JPL

def query_jpl_horizons(identifier, epochs, location):
    """Ask JPL Horizons about one object at the given times 
    (a list of astropy times, normally the start and the end of the exposure).
    Return H, the positions (RA, DEC in degrees) at the first and last times, 
    the plane-of-sky 3-sigma uncertainty ellipse and the rates at the first time.
    """
    jds = [float(t.jd) for t in epochs]
    if jds[0] == jds[-1]:
        jds = jds[:1]
    ephemeris = Horizons(
        id=identifier,
        id_type="smallbody",
        location=location,
        epochs=jds,
    ).ephemerides()
    if len(ephemeris) == 0:
        raise ValueError("Horizons returned no rows")
    last = len(ephemeris)-1

    theta = _table_value(ephemeris, "Theta_3sigma", u.deg)
    # Horizons Theta is measured from increasing RA toward increasing DEC.
    # Convert it to the usual astronomical position angle: east of north.
    lov_pa = None if theta is None else (90.0 - theta) % 180.0
    h_value = _table_value(ephemeris, "H", u.mag)
    major = _table_value(ephemeris, "SMAA_3sigma", u.arcsec)
    minor = _table_value(ephemeris, "SMIA_3sigma", u.arcsec)

    return {
        "H": h_value,
        "SMAA_3sigma": major,
        "SMIA_3sigma": minor,
        "THETA_3sigma": theta,
        "LOV_PA": lov_pa,
        "RA": _table_value(ephemeris, "RA", u.deg, row =0),
        "DEC": _table_value(ephemeris, "DEC", u.deg, row=0),
        "RA_end": _table_value(ephemeris, "RA", u.deg, row=last),
        "DEC_end": _table_value(ephemeris, "DEC", u.deg, row = last),
        "RA_rate": _rate_arcsec_per_min(ephemeris, "RA_rate", row = 0),
        "DEC_rate": _rate_arcsec_per_min(ephemeris, "DEC_rate", row = 0),
    }


JPL_COLUMNS = (
    ("H", u.mag),
    ("SMAA_3sigma", u.arcsec),
    ("SMIA_3sigma", u.arcsec),
    ("THETA_3sigma", u.deg),
    ("LOV_PA", u.deg),
)


def _describe(jpl_data):
    """Short text summarizing what JPL returned, for the progress log."""
    h_text = (f"{jpl_data['H']:.3f}" if jpl_data["H"] is not None
              else "not available")
    if all(jpl_data[k] is not None for k in ("SMAA_3sigma", "SMIA_3sigma","THETA_3sigma", "LOV_PA")):
        ellipse_text = (
            f"3-sigma ellipse = {jpl_data['SMAA_3sigma']:.3f} x "
            f"{jpl_data['SMIA_3sigma']:.3f} arcsec, "
            f"THETA_3sigma = {jpl_data['THETA_3sigma']:.3f} deg,"
            f"LOV PA = {jpl_data['LOV_PA']:.3f} deg"
        )
    else:
        ellipse_text = "uncertainty ellipse not available"
    return f"H = {h_text}; {ellipse_text}"


def fetch_jpl_data(table, start_obs, end_obs, observatory_code):
    """Add H, SMAA_3sigma, SMIA_3sigma, THETA_3sigma and LOV_PA columns, filled from JPL.
    The ellipse is the one at the start of the exposure and is assumed to stay the same until the end.
    Adds RA_end, DEC_end and replaces RA, DEC, RA_rate, DEC_rate and sky_speed with
    Horizons values when available. Otherwise, the identification-catalog position
    and rate are kept, and the end position is calculated from the rate and exposure.
    """
    n = len(table)
    for name, unit in JPL_COLUMNS:
        table[name] = np.full(n, np.nan) * unit

    # JPL-ID adapter change: retain the source and original identification positions.
    discovery_source = (
        str(table["discovery_source"][0])
        if n and "discovery_source" in table.colnames
        else "SkyBot"
    )
    ra0 = u.Quantity(table["RA"]).to_value(u.deg)
    dec0 = u.Quantity(table["DEC"]).to_value(u.deg)
    table["RA_identification"] = ra0 * u.deg
    table["DEC_identification"] = dec0 * u.deg
    if discovery_source == "SkyBot":
        table["RA_skybot"] = ra0 * u.deg
        table["DEC_skybot"] = dec0 * u.deg
    exposure_s = float((end_obs-start_obs).sec)
    ra_end, dec_end = _end_from_rates(ra0, dec0,
                                       u.Quantity(table["RA_rate"]).to_value(u.arcsec/u.min),
                                       u.Quantity(table["DEC_rate"]).to_value(u.arcsec/u.min),
                                       exposure_s)
    table["RA_end"] = ra_end * u.deg
    table["DEC_end"] = dec_end * u.deg
    table["source"] = np.full(n, f"{discovery_source} + rates", dtype="U40")

    print(f"\nFound {n} objects. Fetching positions, H and uncertainty ellipses from JPL Horizons...")

    cache = {}
    n_jpl = 0
    for i in range(n):
        object_name = str(table["Name"][i]).strip()
        print(f" [{i + 1}/{n}] {object_name}: querying Horizons...", flush=True)
        jpl_data = None
        last_error = None

        for identifier in jpl_identifiers(table, i):
            try:
                if identifier not in cache:
                    cache[identifier] = query_jpl_horizons(identifier, [start_obs, end_obs], observatory_code)
                jpl_data = cache[identifier]
                break
            except Exception as exc:
                last_error = exc

        if jpl_data is not None:
            for name, _ in JPL_COLUMNS:
                if jpl_data[name] is not None:
                    # The destination columns already carry their units.
                    table[name][i] = jpl_data[name]
            if all(jpl_data[k] is not None for k in ("RA", "DEC", "RA_end", "DEC_end")):
                for key in ("RA", "DEC", "RA_end", "DEC_end"):
                    table[key][i] = jpl_data[key]
                if jpl_data["RA_rate"] is not None and jpl_data["DEC_rate"] is not None:
                    table["RA_rate"][i] = jpl_data["RA_rate"]
                    table["DEC_rate"][i] = jpl_data["DEC_rate"]
                    table["sky_speed"][i] = np.hypot(jpl_data["RA_rate"], jpl_data["DEC_rate"])
                    table["source"][i] = "JPL Horizons"
                    n_jpl += 1
            print(f" [{i + 1}/{n}] {object_name}: JPL Horizons: {_describe(jpl_data)}")
        elif last_error is not None:
            print(f" [{i + 1}/{n}] {object_name}: JPL query failed ({last_error})")
        else:
            print(f" [{i + 1}/{n}] {object_name}: no usable identifier was available for the JPL query")
    print(f"Horizons supplied complete positions and rates for {n_jpl} of {n} candidates.")
