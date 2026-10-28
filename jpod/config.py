"""Settings that rarely change between runs"""

from pathlib import Path
import astropy.units as u

# The cone search is widened by this margin, so that objects just outside the image (but whose uncertainty ellipse
# reaches it) are not missed. It should be larger than the distance covered by an object during the exposure. 
MARGIN = 5 * u.arcmin   

# Columns of the results table, in this order
OUTPUT_COLUMNS = [
    "Name", "class", "sub-class","source", "RA", "DEC", "RA_end", "DEC_end", "posunc", "V", "H",
    "RA_rate", "DEC_rate", "sky_speed", "trail_length", "trail_length [px]",
    "SMAA_3sigma", "SMIA_3sigma", "LOV_PA", "ellipse_angle",
    "P_pos", "P_mag", "P_det", "detectable",
]

# Figures
ZOOM_HALF_SIZE = 60

# MPC list of observatory codes, used only if the header has no code
PROJECT_DIR = Path(__file__).resolve().parent
OBSCODES_JSON = PROJECT_DIR / "fits_header" / "obscodes_extended.json"