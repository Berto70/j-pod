"""Shared units and table conventions for asteroid discovery providers."""

import numpy as np
import astropy.units as u
from astropy.table import QTable


RA_RATE_COLUMN = "RA_cosdec_rate"
DEC_RATE_COLUMN = "DEC_rate"
RATE_UNIT = u.arcsec / u.hour


def _quantity(values, unit):
    """Convert provider values to a quantity in the shared unit."""
    if isinstance(values, u.Quantity):
        return values.to(unit)
    return np.asarray(values) * unit


def sky_speed(ra_cosdec_rate, dec_rate):
    """Return speed from east/north tangent-plane components.

    The RA component already includes cos(DEC), so applying another cosine
    here would incorrectly reduce the eastward motion a second time.
    """
    return np.hypot(ra_cosdec_rate, dec_rate)


def make_provider_table(
    *,
    names,
    numbers,
    ra,
    dec,
    ra_cosdec_rate,
    dec_rate,
    magnitudes,
    position_uncertainties,
    classes,
    subclasses,
    provider,
    epoch,
):
    """Build the common provider table with explicit units and missing values."""
    table = QTable()
    table["Name"] = np.asarray(names, dtype=str)
    table["Number"] = np.asarray(numbers, dtype=str)
    table["RA"] = _quantity(ra, u.deg)
    table["DEC"] = _quantity(dec, u.deg)
    table[RA_RATE_COLUMN] = _quantity(ra_cosdec_rate, RATE_UNIT)
    table[DEC_RATE_COLUMN] = _quantity(dec_rate, RATE_UNIT)
    table["sky_speed"] = sky_speed(
        table[RA_RATE_COLUMN], table[DEC_RATE_COLUMN]
    ).to(RATE_UNIT)
    table["posunc"] = _quantity(position_uncertainties, u.arcsec)
    table["V"] = _quantity(magnitudes, u.mag)
    table["class"] = np.asarray(classes, dtype=str)
    table["sub-class"] = np.asarray(subclasses, dtype=str)
    table["provider"] = np.full(len(table), provider, dtype=f"U{len(provider)}")
    table["provider_epoch"] = np.full(len(table), epoch.utc.isot, dtype="U32")
    table.meta["provider"] = provider
    table.meta["provider_epoch_jd"] = float(epoch.utc.jd)
    return table
