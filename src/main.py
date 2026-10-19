"""Find the asteroids in a FITS image with SkyBoT and add JPL uncertainty ellipses."""

import astropy.units as u
from astropy.coordinates import SkyCoord
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from catalogs import fetch_jpl_data, query_skybot
from image_geometry import (
    add_ellipse_image_angle,
    filter_in_image,
    image_orientation,
    load_image,
    search_radius,
)
from plotting import plot_field
from detection import add_detection_probability
from output import save_table_csv

from time import time

# ---------------------------------------------------------------- settings
FITS_FILE = "ztf_20210820142419_000526_zr_c14_o_q3_scimrefdiffimg.fits"
OBSERVATORY_CODE = "I41" # I41 for ZTF, 675 for Palomar
MARGIN = 5 * u.arcmin    # widen the cone so ellipses of nearby objects are not missed
MAGNIFY = 10             # ellipse magnification in the plot (1 = true size)
V_LIM = 15
SIGMA_V = 0.5
P_MIN = 0.05
SAVE_FIGURES = True
PRINT_TABLE = False
PREFIX = Path(FITS_FILE).stem

OUTPUT_COLUMNS = [
    "Name", "RA", "DEC", "posunc", "V", "H",
    "RA_rate", "DEC_rate", "sky_speed",
    "SMAA_3sigma", "SMIA_3sigma", "LOV_PA", "ellipse_angle",
    "P_pos", "P_mag", "P_det",
]


def main():
    # 1. Image and orientation
    info = load_image(FITS_FILE)
    angle_north, angle_east, parity = image_orientation(info.wcs, info.center)
    print(
        f"At the image center: North points at {angle_north:.4f} deg, "
        f"East at {angle_east:.4f} deg (CCW from +x, 0 = right, 90 = up); "
        f"parity = {parity:+d}"
    )
    time_in = time()
    # 2. SkyBoT cone search
    field = SkyCoord(info.center.ra.deg * u.deg, info.center.dec.deg * u.deg, frame="icrs")
    radius = search_radius(info, MARGIN)
    print(f"Querying SkyBoT for asteroids in the image field in a radius of {radius}...")
    table = query_skybot(field, radius, info.epoch, OBSERVATORY_CODE)
    if len(table) == 0:
        print("No asteroids found in this field of view.")
        return
    time_out = time()
    print("SkyBot cone done", time_out-time_in)

    time_in = time()
    # 3. JPL data, ellipse orientation in the image, overlap filter
    fetch_jpl_data(table, info.epoch, OBSERVATORY_CODE)
    add_ellipse_image_angle(table, info.wcs)
    table = filter_in_image(table, info)
    print(f"\n{len(table)} objects overlap the image.")
    if len(table) == 0:
        return
    time_out = time()
    print("JPL data done", time_out-time_in)

    # 4. Detection probability and survivors
    add_detection_probability(table, info, V_LIM, SIGMA_V)
    survivors = np.asarray(table["P_det"],dtype=float)>=P_MIN
    n_surv = int(survivors.sum())
    print(f"{n_surv} of {len(table)} objects have P_det >= {P_MIN}"
          f"(V_lim = {V_LIM}, sigma_V = {SIGMA_V})")
    print(f"Expected number of detections: {np.nansum(table["P_det"]):.2f}")


    # 5. Plots
    plot_field(
        info,
        table, 
        angle_north,
        angle_east, 
        magnify=MAGNIFY,
        show = False, 
        title =f"All objects overlapping the image (ellipses x{MAGNIFY})", 
        output_file=f"{PREFIX}_candidates.png" if SAVE_FIGURES else None,
        )
    plot_field(
        info, 
        table, 
        angle_north, 
        angle_east, 
        magnify=MAGNIFY, 
        show = False, 
        survivors=survivors, 
        title = (f"Green: P_det >= {P_MIN} (V_lim ={V_LIM});"
                f"grey: below threshold (ellipses x{MAGNIFY})"),
        output_file=f"{PREFIX}_survivors.png" if SAVE_FIGURES else None,    
        )
    plt.show()

    # 6. Table
    csv_file = f"{PREFIX}_results.csv"
    save_table_csv(table, csv_file, OUTPUT_COLUMNS)
    print(f"\nTable saved to {csv_file}")
    if PRINT_TABLE:
        print("\n--- Complete target results ---")
        table[OUTPUT_COLUMNS].pprint_all()


if __name__ == "__main__":
    main()
