"""Asteroid discovery using the IMCCE SkyBoT cone-search service."""

import numpy as np
import astropy.units as u
from astropy.time import Time
from astroquery.imcce import Skybot

from .schema import make_provider_table


def _split_type(values):
    standalone_classes = {
        "Trojan": ("Trojan", ""),
        "Mars-Crosser": ("MC", ""),
        "Hungaria": ("MB", "Hungaria"),
        "Comet": ("Comet", ""),
    }
    classes = []
    subclasses = []
    unknown_types = set()

    for value in values:
        if value is None or np.ma.is_masked(value) or str(value).strip() == "":
            classes.append("")
            subclasses.append("")
            continue
        object_type = str(value).strip()
        if ">" in object_type:
            main_class, subclass = object_type.split(">", 1)
        elif object_type in standalone_classes:
            main_class, subclass = standalone_classes[object_type]
        else:
            main_class, subclass = object_type, ""
            unknown_types.add(object_type)
        classes.append(main_class)
        subclasses.append(subclass)

    if unknown_types:
        print("\nUnmapped standalone dynamical classes:")
        for value in sorted(unknown_types):
            print(f" - {value}")
        print()
    return classes, subclasses


def _rate_column(table):
    for name in ("RAcosD_rate", "RA_rate"):
        if name in table.colnames:
            return name
    raise KeyError("SkyBoT result has no RA motion-rate column")


def query_skybot(field, radius, epoch, observatory_code):
    """Query asteroids and comets and normalize them to the shared schema."""
    epoch = Time(epoch).utc
    results = Skybot.cone_search(
        field,
        u.Quantity(radius).to(u.arcsec),
        epoch,
        location=observatory_code,
        # MPChecker reports minor planets, so include asteroids and comets only.
        find_planets=False,
        find_asteroids=True,
        find_comets=True,
        # SkyBoT's default is 120 arcsec; zero disables this positional-error cut.
        position_error=0 * u.arcsec,
    )
    classes, subclasses = _split_type(
        results["Type"] if "Type" in results.colnames else [""] * len(results)
    )
    numbers = [
        "" if np.ma.is_masked(value) or str(value).strip() in {"", "-1"}
        else str(value).strip()
        for value in results["Number"]
    ]

    table = make_provider_table(
        names=results["Name"],
        numbers=numbers,
        ra=results["RA"],
        dec=results["DEC"],
        ra_cosdec_rate=results[_rate_column(results)],
        dec_rate=results["DEC_rate"],
        magnitudes=results["V"],
        position_uncertainties=results["posunc"],
        classes=classes,
        subclasses=subclasses,
        provider="SkyBoT",
        epoch=epoch,
    )
    print(f"SkyBoT found {len(table)} small bodies.", flush=True)
    return table
