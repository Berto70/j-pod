"""Live, end-to-end comparison of MPC MPChecker and SkyBoT on the J75 image.

This is intentionally an opt-in integration test: it contacts MPChecker,
SkyBoT, and JPL Horizons, and writes a comparison CSV under output/tables.
Run it from the repository root with:

    JPOD_RUN_J75_PROVIDER_INTEGRATION=1 \
        PYTHONPATH=. .venv/bin/python -m unittest discover \
        -s tests -p test_providers.py -v

The CSV has one row per object found by either provider. Objects are joined
using their normalized designation, which avoids treating an MPC number prefix
as a different object name. Each provider's measurements are kept in separate
columns. Horizons position/rate values are reported at that provider's search
epoch; H and the uncertainty ellipse are reported at exposure start.
"""

import csv
import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import astropy.units as u
import numpy as np

import jpod.main as main_module
from jpod.config import OBSCODES_JSON
from jpod.image_geometry import load_image, search_radius


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
J75_IMAGE = REPOSITORY_ROOT / "data" / "J75_01_20110720204510.fits"
COMPARISON_CSV = (
    REPOSITORY_ROOT / "output" / "tables" / "J75_provider_comparison.csv"
)
RUN_LIVE_INTEGRATION = (
    os.environ.get("JPOD_RUN_J75_PROVIDER_INTEGRATION") == "1"
)


def _column_value(table, column_name, row, unit=None):
    """Return a CSV-safe cell, converting quantities and blanking missing data."""
    if column_name not in table.colnames:
        return ""

    value = table[column_name][row]
    if np.ma.is_masked(value) or value is None:
        return ""

    if unit is not None and hasattr(value, "to_value"):
        value = value.to_value(unit)

    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return ""
    return value.item() if isinstance(value, np.generic) else value


def _object_key(table, row):
    """Normalize designations that providers format with different prefixes."""
    name = str(table["Name"][row]).strip().lower()
    # MPChecker may return "(12345) 2011 AB" while SkyBoT returns "2011 AB".
    # The designation identifies the same object without relying on the
    # provider-specific number formatting.
    name = re.sub(r"^\(\s*\d+\s*\)\s*", "", name)
    normalized_name = re.sub(r"[^a-z0-9]+", "", name)
    if normalized_name:
        return f"name:{normalized_name}"

    number = str(table["Number"][row]).strip()
    return f"number:{number}"


def _comparison_columns():
    """List all output fields in human-readable order, including their units."""
    shared = [
        "Object key",
        "MPC number",
        "Observatory code",
        "Image exposure start (UTC)",
        "Image exposure end (UTC)",
        "Search radius (deg)",
    ]
    provider_fields = [
        ("Name", ""),
        ("Class", ""),
        ("Subclass", ""),
        ("Search epoch (UTC)", ""),
        ("RA at search epoch", "deg"),
        ("DEC at search epoch", "deg"),
        ("RA*cos(DEC) rate at search epoch", "arcsec/hour"),
        ("DEC rate at search epoch", "arcsec/hour"),
        ("Sky speed at search epoch", "arcsec/hour"),
        ("V magnitude", "mag"),
        ("Position uncertainty", "arcsec"),
        ("Horizons comparison epoch (UTC)", ""),
        ("Horizons RA at comparison epoch", "deg"),
        ("Horizons DEC at comparison epoch", "deg"),
        ("Horizons RA*cos(DEC) rate at comparison epoch", "arcsec/hour"),
        ("Horizons DEC rate at comparison epoch", "arcsec/hour"),
        ("Horizons sky speed at comparison epoch", "arcsec/hour"),
        ("Horizons H at exposure start", "mag"),
        ("Horizons semi-major uncertainty axis at exposure start", "arcsec"),
        ("Horizons semi-minor uncertainty axis at exposure start", "arcsec"),
        ("Horizons LOV position angle at exposure start", "deg"),
        ("Working-data source", ""),
    ]
    columns = list(shared)
    for label in ("MPC MPChecker", "SkyBoT"):
        columns.extend(
            f"{label} {field}" + (f" [{unit}]" if unit else "")
            for field, unit in provider_fields
        )
    return columns


def _provider_cells(table, row):
    """Return the raw provider values and same-epoch Horizons comparison values."""
    values = {
        "Name": _column_value(table, "Name", row),
        "Class": _column_value(table, "class", row),
        "Subclass": _column_value(table, "sub-class", row),
        "Search epoch (UTC)": _column_value(table, "provider_epoch", row),
        "RA at search epoch": _column_value(
            table, "RA_provider", row, u.deg
        ),
        "DEC at search epoch": _column_value(
            table, "DEC_provider", row, u.deg
        ),
        "RA*cos(DEC) rate at search epoch": _column_value(
            table, "RA_cosdec_rate_provider", row, u.arcsec / u.hour
        ),
        "DEC rate at search epoch": _column_value(
            table, "DEC_rate_provider", row, u.arcsec / u.hour
        ),
        "Sky speed at search epoch": _column_value(
            table, "sky_speed_provider", row, u.arcsec / u.hour
        ),
        "V magnitude": _column_value(table, "V_provider", row, u.mag),
        "Position uncertainty": _column_value(
            table, "posunc_provider", row, u.arcsec
        ),
        "Horizons comparison epoch (UTC)": _column_value(
            table, "horizons_epoch", row
        ),
        "Horizons RA at comparison epoch": _column_value(
            table, "RA_horizons", row, u.deg
        ),
        "Horizons DEC at comparison epoch": _column_value(
            table, "DEC_horizons", row, u.deg
        ),
        "Horizons RA*cos(DEC) rate at comparison epoch": _column_value(
            table, "RA_cosdec_rate_horizons", row, u.arcsec / u.hour
        ),
        "Horizons DEC rate at comparison epoch": _column_value(
            table, "DEC_rate_horizons", row, u.arcsec / u.hour
        ),
        "Horizons sky speed at comparison epoch": _column_value(
            table, "sky_speed_horizons", row, u.arcsec / u.hour
        ),
        "Horizons H at exposure start": _column_value(
            table, "H", row, u.mag
        ),
        "Horizons semi-major uncertainty axis at exposure start": _column_value(
            table, "SMAA_3sigma", row, u.arcsec
        ),
        "Horizons semi-minor uncertainty axis at exposure start": _column_value(
            table, "SMIA_3sigma", row, u.arcsec
        ),
        "Horizons LOV position angle at exposure start": _column_value(
            table, "LOV_PA", row, u.deg
        ),
        "Working-data source": _column_value(table, "source", row),
    }
    return values


def _write_comparison_csv(provider_tables, image_info, radius, output_path):
    """Join both pipeline tables and write provider/Horizons values side by side."""
    # Each dictionary entry is keyed by asteroid number when possible. Using
    # the union keeps objects found by only one service visible in the report.
    combined = {}
    unit_by_field = {
        "RA at search epoch": "deg",
        "DEC at search epoch": "deg",
        "RA*cos(DEC) rate at search epoch": "arcsec/hour",
        "DEC rate at search epoch": "arcsec/hour",
        "Sky speed at search epoch": "arcsec/hour",
        "V magnitude": "mag",
        "Position uncertainty": "arcsec",
        "Horizons RA at comparison epoch": "deg",
        "Horizons DEC at comparison epoch": "deg",
        "Horizons RA*cos(DEC) rate at comparison epoch": "arcsec/hour",
        "Horizons DEC rate at comparison epoch": "arcsec/hour",
        "Horizons sky speed at comparison epoch": "arcsec/hour",
        "Horizons H at exposure start": "mag",
        "Horizons semi-major uncertainty axis at exposure start": "arcsec",
        "Horizons semi-minor uncertainty axis at exposure start": "arcsec",
        "Horizons LOV position angle at exposure start": "deg",
    }
    for provider_name, table in provider_tables.items():
        prefix = "MPC MPChecker" if provider_name == "MPC MPChecker" else "SkyBoT"
        for row in range(len(table)):
            key = _object_key(table, row)
            entry = combined.setdefault(
                key,
                {
                    "Object key": key,
                    "MPC number": _column_value(table, "Number", row),
                    "Observatory code": image_info.observatory_code,
                    "Image exposure start (UTC)": image_info.start_obs.utc.isot,
                    "Image exposure end (UTC)": image_info.end_obs.utc.isot,
                    "Search radius (deg)": radius.to_value(u.deg),
                },
            )
            for field, value in _provider_cells(table, row).items():
                # The unit suffix in each header makes raw numbers
                # self-describing when the CSV is opened outside Astropy.
                unit = unit_by_field.get(field)
                column = f"{prefix} {field}" + (f" [{unit}]" if unit else "")
                entry[column] = value

    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = _comparison_columns()
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(combined.values())
    return combined


@unittest.skipUnless(
    RUN_LIVE_INTEGRATION,
    "Set JPOD_RUN_J75_PROVIDER_INTEGRATION=1 to run live provider queries.",
)
class J75ProviderComparisonIntegrationTest(unittest.TestCase):
    """Exercise the real image pipeline once with each live discovery provider."""

    def test_j75_image_comparison_csv_contains_both_providers_and_horizons(self):
        # Fail early with a useful message if this test is run outside the
        # repository checkout or if the sample image has been moved.
        self.assertTrue(J75_IMAGE.is_file(), f"J75 FITS image not found: {J75_IMAGE}")

        # Load the image once to record the exact shared observation window and
        # search radius used in both pipeline runs.
        image_info = load_image(
            str(J75_IMAGE), None, None, None, OBSCODES_JSON
        )
        radius = search_radius(image_info, main_module.MARGIN)
        self.assertEqual(image_info.observatory_code, "J75")

        # Intercept the normal compact-CSV writer so the two pipeline runs
        # cannot overwrite one another. The full in-memory enriched tables are
        # copied here and later written together as a comparison-only CSV.
        pipeline_tables = {}

        def capture_full_table(table, _output_dir, _filename, _columns):
            provider_name = str(table["provider"][0])
            pipeline_tables[provider_name] = table.copy(copy_data=True)

        # Plot rendering is unrelated to provider comparison, so replace those
        # functions while keeping image loading, searches, Horizons enrichment,
        # geometric filtering, trail calculation, and probability calculation
        # in the actual production pipeline.
        with tempfile.TemporaryDirectory(prefix="jpod-j75-") as temp_dir:
            with (
                patch.object(
                    main_module, "save_table_csv", side_effect=capture_full_table
                ),
                patch.object(main_module, "plot_field"),
                patch.object(main_module, "plot_cutouts"),
            ):
                for provider_argument in ("mpc", "skybot"):
                    args = main_module.create_parser().parse_args(
                        ["--provider", provider_argument]
                    )
                    main_module.pipeline(
                        str(J75_IMAGE), args, Path(temp_dir)
                    )

        # The integration is meaningful only if both complete pipelines found
        # in-image objects and ran through the save step.
        self.assertIn("MPC MPChecker", pipeline_tables)
        self.assertIn("SkyBoT", pipeline_tables)
        self.assertGreater(len(pipeline_tables["MPC MPChecker"]), 0)
        self.assertGreater(len(pipeline_tables["SkyBoT"]), 0)

        # Save a separate wide CSV so original MPC, SkyBoT, and same-epoch
        # Horizons values can be compared without changing the app's normal,
        # intentionally compact result CSV.
        comparison_rows = _write_comparison_csv(
            pipeline_tables, image_info, radius, COMPARISON_CSV
        )
        self.assertTrue(COMPARISON_CSV.is_file())
        self.assertGreater(len(comparison_rows), 0)

        # Confirm both provider columns, both search epochs, and Horizons
        # comparison columns are actually populated in at least one row.
        self.assertTrue(
            any(
                row.get("MPC MPChecker RA at search epoch [deg]") != ""
                and row.get("SkyBoT RA at search epoch [deg]") != ""
                for row in comparison_rows.values()
            ),
            "No asteroid number/name was matched across both provider results.",
        )
        for table in pipeline_tables.values():
            self.assertIn("RA_horizons", table.colnames)
            self.assertIn("provider_epoch", table.colnames)
            self.assertIn("horizons_epoch", table.colnames)
