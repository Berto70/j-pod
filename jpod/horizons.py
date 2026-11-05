"""JPL Horizons queries and enrichment of provider asteroid tables."""

import re

import astropy.units as u
import numpy as np
from astropy.time import Time
from astroquery.jplhorizons import Horizons

from .providers.schema import DEC_RATE_COLUMN, RA_RATE_COLUMN, RATE_UNIT, sky_speed


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
            if re.fullmatch(r"\d+\.0+", number):
                number = number.split(".", 1)[0]
            if number.isdigit():
                identifiers.append(number)

    name = _usable_text(table["Name"][row])
    if name:
        wrapped_designation = re.fullmatch(
            r"\(\s*(\d{4}\s+[A-Z]{1,2}\d+[A-Z]?)\s*\)", name
        )
        if wrapped_designation:
            name = wrapped_designation.group(1)
        match = re.match(r"^\(\s*(\d+)\s*\)", name)
        if match:
            identifiers.append(match.group(1))
        identifiers.append(name)
    return list(dict.fromkeys(identifiers))


def _table_value(table, column, unit=None, row=0):
    """Return one finite scalar from a Horizons table, converted to ``unit``."""
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


def _rate_arcsec_per_hour(table, column, row):
    """Read a Horizons angular rate in the shared arcsec/hour convention."""
    if column not in table.colnames or row >= len(table):
        return None
    column_data = table[column]
    value = column_data[row]
    if value is None or np.ma.is_masked(value):
        return None
    unit = getattr(column_data, "unit", None) or RATE_UNIT
    try:
        quantity = value if hasattr(value, "to_value") else value * unit
        result = float(quantity.to_value(RATE_UNIT))
    except (TypeError, ValueError, u.UnitConversionError):
        return None
    return result if np.isfinite(result) else None


def _row_for_epoch(epochs, epoch):
    """Find a queried Horizons row corresponding to an exact requested epoch."""
    for row, queried_epoch in enumerate(epochs):
        if abs((queried_epoch - epoch).to_value(u.s)) < 1e-5:
            return row
    raise ValueError(f"Horizons query did not include epoch {epoch.utc.isot}")


def _rates_at(ephemeris, row):
    ra_rate_column = "RA_rate" if "RA_rate" in ephemeris.colnames else "RAcosD_rate"
    ra_rate = _rate_arcsec_per_hour(ephemeris, ra_rate_column, row)
    dec_rate = _rate_arcsec_per_hour(ephemeris, "DEC_rate", row)
    # Horizons' RA rate is already the eastward tangent-plane component.
    speed = (
        float(
            sky_speed(ra_rate * RATE_UNIT, dec_rate * RATE_UNIT).to_value(RATE_UNIT)
        )
        if ra_rate is not None and dec_rate is not None
        else None
    )
    return ra_rate, dec_rate, speed


def _positions_at(ephemeris, row):
    return (
        _table_value(ephemeris, "RA", u.deg, row),
        _table_value(ephemeris, "DEC", u.deg, row),
    )


def query_jpl_horizons(identifier, epochs, location):
    """Query Horizons for comparison-epoch and exposure endpoint values.

    The first epoch is the provider's discovery epoch; the next two are the
    image exposure start and end. Unique epochs are sent in one Horizons query.
    """
    unique_epochs = []
    for epoch in epochs:
        epoch = Time(epoch).utc
        if not any(abs((epoch - present).to_value(u.s)) < 1e-5
                   for present in unique_epochs):
            unique_epochs.append(epoch)
    # MPChecker rounds its epoch, which can move it slightly after the image time.
    unique_epochs.sort(key=lambda epoch: float(epoch.jd))

    ephemeris = Horizons(
        id=identifier,
        id_type="smallbody",
        location=location,
        epochs=[float(epoch.jd) for epoch in unique_epochs],
    ).ephemerides()
    if len(ephemeris) == 0:
        raise ValueError("Horizons returned no rows")

    comparison_row = _row_for_epoch(unique_epochs, Time(epochs[0]).utc)
    start_row = _row_for_epoch(unique_epochs, Time(epochs[1]).utc)
    end_row = _row_for_epoch(unique_epochs, Time(epochs[2]).utc)
    comparison_ra, comparison_dec = _positions_at(ephemeris, comparison_row)
    start_ra, start_dec = _positions_at(ephemeris, start_row)
    end_ra, end_dec = _positions_at(ephemeris, end_row)
    comparison_ra_rate, comparison_dec_rate, comparison_speed = _rates_at(
        ephemeris, comparison_row
    )
    start_ra_rate, start_dec_rate, start_speed = _rates_at(ephemeris, start_row)

    theta = _table_value(ephemeris, "Theta_3sigma", u.deg, start_row)
    lov_pa = None if theta is None else (90.0 - theta) % 180.0
    return {
        "H": _table_value(ephemeris, "H", u.mag, start_row),
        "SMAA_3sigma": _table_value(ephemeris, "SMAA_3sigma", u.arcsec, start_row),
        "SMIA_3sigma": _table_value(ephemeris, "SMIA_3sigma", u.arcsec, start_row),
        "THETA_3sigma": theta,
        "LOV_PA": lov_pa,
        "RA": start_ra,
        "DEC": start_dec,
        "RA_end": end_ra,
        "DEC_end": end_dec,
        RA_RATE_COLUMN: start_ra_rate,
        DEC_RATE_COLUMN: start_dec_rate,
        "sky_speed": start_speed,
        "RA_horizons": comparison_ra,
        "DEC_horizons": comparison_dec,
        "RA_cosdec_rate_horizons": comparison_ra_rate,
        "DEC_rate_horizons": comparison_dec_rate,
        "sky_speed_horizons": comparison_speed,
    }


JPL_COLUMNS = (
    ("H", u.mag),
    ("SMAA_3sigma", u.arcsec),
    ("SMIA_3sigma", u.arcsec),
    ("THETA_3sigma", u.deg),
    ("LOV_PA", u.deg),
)

COMPARISON_COLUMNS = (
    ("RA_cosdec_rate_provider", RATE_UNIT),
    ("DEC_rate_provider", RATE_UNIT),
    ("sky_speed_provider", RATE_UNIT),
    ("RA_horizons", u.deg),
    ("DEC_horizons", u.deg),
    ("RA_cosdec_rate_horizons", RATE_UNIT),
    ("DEC_rate_horizons", RATE_UNIT),
    ("sky_speed_horizons", RATE_UNIT),
)


def _describe(jpl_data):
    """Summarize Horizons values for the progress log."""
    h_text = (
        f"{jpl_data['H']:.3f}" if jpl_data["H"] is not None else "not available"
    )
    if all(
        jpl_data[key] is not None
        for key in ("SMAA_3sigma", "SMIA_3sigma", "THETA_3sigma", "LOV_PA")
    ):
        ellipse_text = (
            f"3-sigma ellipse = {jpl_data['SMAA_3sigma']:.3f} x "
            f"{jpl_data['SMIA_3sigma']:.3f} arcsec, "
            f"THETA_3sigma = {jpl_data['THETA_3sigma']:.3f} deg, "
            f"LOV PA = {jpl_data['LOV_PA']:.3f} deg"
        )
    else:
        ellipse_text = "uncertainty ellipse not available"
    return f"H = {h_text}; {ellipse_text}"


def enrich_with_horizons(table, start_obs, end_obs, observatory_code):
    """Keep provider values, add comparable Horizons values, and enrich the table.

    The un-suffixed position and rate columns remain the values used by the
    image pipeline: Horizons values when available, otherwise provider values.
    Provider and Horizons comparison columns are preserved independently.
    """
    if len(table) == 0:
        return table

    provider = str(table["provider"][0])
    provider_epoch = Time(float(table.meta["provider_epoch_jd"]), format="jd", scale="utc")
    observation_start = Time(start_obs).utc
    observation_end = Time(end_obs).utc

    # Keep the original discovery measurements for provider-versus-Horizons
    # comparisons. The normal CSV remains compact because OUTPUT_COLUMNS
    # deliberately excludes these diagnostic columns.
    for name in (
        "RA",
        "DEC",
        "V",
        "posunc",
        RA_RATE_COLUMN,
        DEC_RATE_COLUMN,
        "sky_speed",
    ):
        table[f"{name}_provider"] = table[name].copy()

    for name, unit in JPL_COLUMNS:
        table[name] = np.full(len(table), np.nan) * unit
    for name, unit in COMPARISON_COLUMNS:
        if name not in table.colnames:
            table[name] = np.full(len(table), np.nan) * unit

    table["RA_end"] = np.full(len(table), np.nan) * u.deg
    table["DEC_end"] = np.full(len(table), np.nan) * u.deg
    table["provider_epoch"] = np.full(
        len(table), provider_epoch.utc.isot, dtype="U32"
    )
    table["horizons_epoch"] = np.full(
        len(table), provider_epoch.utc.isot, dtype="U32"
    )
    table["observation_epoch"] = np.full(
        len(table), observation_start.utc.isot, dtype="U32"
    )
    table["source"] = np.full(len(table), provider, dtype="U40")

    exposure_seconds = float((observation_end - observation_start).to_value(u.s))
    fallback_ra_end, fallback_dec_end = _end_from_rates(
        table["RA"].to_value(u.deg),
        table["DEC"].to_value(u.deg),
        table[RA_RATE_COLUMN].to_value(RATE_UNIT),
        table[DEC_RATE_COLUMN].to_value(RATE_UNIT),
        exposure_seconds,
    )
    table["RA_end"] = fallback_ra_end * u.deg
    table["DEC_end"] = fallback_dec_end * u.deg

    print(
        f"\nFound {len(table)} objects. Fetching positions, H and uncertainty "
        "ellipses from JPL Horizons..."
    )
    cache = {}
    n_jpl = 0
    for row in range(len(table)):
        object_name = str(table["Name"][row]).strip()
        jpl_data = None
        last_error = None
        epochs = [provider_epoch, observation_start, observation_end]

        for identifier in jpl_identifiers(table, row):
            try:
                if identifier not in cache:
                    cache[identifier] = query_jpl_horizons(
                        identifier, epochs, observatory_code
                    )
                jpl_data = cache[identifier]
                break
            except Exception as exc:
                last_error = exc

        if jpl_data is None:
            if last_error is not None:
                print(f" [{row + 1}/{len(table)}] {object_name}: JPL query failed ({last_error})")
            else:
                print(
                    f" [{row + 1}/{len(table)}] {object_name}: "
                    "no usable identifier was available for the JPL query"
                )
            continue

        for name, unit in JPL_COLUMNS:
            if jpl_data[name] is not None:
                table[name][row] = jpl_data[name] * unit
        for name, unit in COMPARISON_COLUMNS:
            value = jpl_data.get(name)
            if value is not None:
                table[name][row] = value * unit

        positions = ("RA", "DEC", "RA_end", "DEC_end")
        has_complete_horizons_track = (
            all(jpl_data[name] is not None for name in positions)
            and jpl_data[RA_RATE_COLUMN] is not None
            and jpl_data[DEC_RATE_COLUMN] is not None
            and jpl_data["sky_speed"] is not None
        )
        if has_complete_horizons_track:
            for name in positions:
                table[name][row] = jpl_data[name] * u.deg
            table[RA_RATE_COLUMN][row] = jpl_data[RA_RATE_COLUMN] * RATE_UNIT
            table[DEC_RATE_COLUMN][row] = jpl_data[DEC_RATE_COLUMN] * RATE_UNIT
            table["sky_speed"][row] = jpl_data["sky_speed"] * RATE_UNIT
            table["source"][row] = "JPL Horizons"
            n_jpl += 1

        print(
            f" [{row + 1}/{len(table)}] {object_name}: "
            f"JPL Horizons: {_describe(jpl_data)}"
        )
    print(
        f"Horizons supplied rates for {n_jpl} of {len(table)} candidates. "
        "Provider and Horizons comparison values are retained in separate columns."
    )
    return table


def _end_from_rates(ra_deg, dec_deg, ra_cosdec_rate, dec_rate, exposure_seconds):
    """Project tangent-plane rates across an exposure to RA/DEC coordinates."""
    exposure = exposure_seconds * u.s
    eastward_change = (
        np.asarray(ra_cosdec_rate) * RATE_UNIT * exposure
    ).to_value(u.deg)
    northward_change = (
        np.asarray(dec_rate) * RATE_UNIT * exposure
    ).to_value(u.deg)
    ra_end = np.asarray(ra_deg) + eastward_change / np.cos(np.radians(dec_deg))
    dec_end = np.asarray(dec_deg) + northward_change
    return ra_end, dec_end
