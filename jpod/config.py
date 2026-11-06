"""Settings that rarely change between runs"""

from pathlib import Path
import astropy.units as u

# The cone search is widened by this margin, so that objects just outside the image (but whose uncertainty ellipse
# reaches it) are not missed. It should be larger than the distance covered by an object during the exposure. 
MARGIN = 5 * u.arcmin   

# Columns of the results table, in this order
OUTPUT_COLUMNS = [
    "Name", "class", "sub-class", "source",
    "RA", "DEC", "RA_end", "DEC_end", "V", "V_trailing", "H",
    "RA_cosdec_rate", "DEC_rate", "sky_speed",
    "trail_length", "trail_length [px]", "trail_length / FWHM",
    "trailing_loss",
    "SMAA_3sigma", "SMIA_3sigma", "LOV_PA", "ellipse_angle",
    "P_pos", "P_mag", "P_det", "detectable",
]

# Figures
ZOOM_HALF_SIZE = 60

# MPC list used to infer an observatory code when no valid code is in the header.
PROJECT_DIR = Path(__file__).resolve().parent
OBSCODES_JSON = PROJECT_DIR / "fits_header" / "obscodes_extended.json"
V50_CATALOG_DIR = PROJECT_DIR.parent / "astap"


def resolve_v50_catalog_dir(workdir: Path) -> Path:
    """Find the local catalog beside the checkout or in the selected workdir."""
    if V50_CATALOG_DIR.is_dir():
        return V50_CATALOG_DIR

    workdir_catalog = workdir / "astap"
    if workdir_catalog.is_dir():
        return workdir_catalog

    return V50_CATALOG_DIR
