"""Find the asteroids in a FITS image with SkyBoT and add JPL uncertainty ellipses."""
import argparse
from pathlib import Path

import astropy.units as u
from astropy.coordinates import SkyCoord
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

# parsing command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dir", default="../data") # FITS directory
parser.add_argument("-f", "--file", default="Null") # FITS file name
parser.add_argument("-outdir", "--output_dir", default="../output") # output directory
parser.add_argument("-o", "--observatory", default="I41") # observatory code 

parser.add_argument("-vlim", "--vlim", type=float, default=21.5) # limiting magnitude
parser.add_argument("-sigv", "--sigma_v", type=float, default=0.5) # uncertainty in magnitude
parser.add_argument("-pmin", "--pmin", type=float, default=0.05) # minimum detection probability
parser.add_argument("-savefig", "--savefig", action="store_true", default=False) # save figures
parser.add_argument("-printtable", "--printtable", action="store_true", default=False) # print table to console

args = parser.parse_args()

FITS_DIR = Path(args.dir)
FITS_FILE = FITS_DIR / args.file
OUTPUT_DIR = Path(args.output_dir)
OUTPUT_DIR_plots = OUTPUT_DIR / "plots"
OUTPUT_DIR_tables = OUTPUT_DIR / "tables"
OBSERVATORY_CODE = args.observatory # I41 for ZTF, 675 for Palomar, Z32 for OAJ

MARGIN = 5 * u.arcmin    # widen the cone so ellipses of nearby objects are not missed
MAGNIFY = 10             # ellipse magnification in the plot (1 = true size)
V_LIM = args.vlim
SIGMA_V = args.sigma_v
P_MIN = args.pmin
SAVE_FIGURES = args.savefig
PRINT_TABLE = args.printtable

OUTPUT_COLUMNS = [
    "Name", "RA", "DEC", "posunc", "V", "H",
    "RA_rate", "DEC_rate", "sky_speed",
    "SMAA_3sigma", "SMIA_3sigma", "LOV_PA", "ellipse_angle",
    "P_pos", "P_mag", "P_det",
]


def pipeline(fits_file: str):

    prefix = Path(fits_file).stem
    info = load_image(fits_file)

    # 1. Image and orientation
    angle_north, angle_east, parity = image_orientation(info.wcs, info.center)
    print(
        f"At the image center: North points at {angle_north:.4f} deg, "
        f"East at {angle_east:.4f} deg (CCW from +x, 0 = right, 90 = up); "
        f"parity = {parity:+d}"
    )

    # 2. SkyBoT cone search
    field = SkyCoord(info.center.ra.deg * u.deg, info.center.dec.deg * u.deg, frame="icrs")
    radius = search_radius(info, MARGIN)
    print(f"Querying SkyBoT for asteroids in the image field in a radius of {radius}...")
    table = query_skybot(field, radius, info.epoch, OBSERVATORY_CODE)
    if len(table) == 0:
        print("No asteroids found in this field of view.")
        return

    # 3. JPL data, ellipse orientation in the image, overlap filter
    fetch_jpl_data(table, info.epoch, OBSERVATORY_CODE)
    add_ellipse_image_angle(table, info.wcs)
    table = filter_in_image(table, info)
    print(f"\n{len(table)} objects overlap the image.")
    if len(table) == 0:
        return

    # 4. Detection probability and survivors
    add_detection_probability(table, info, V_LIM, SIGMA_V)
    survivors = np.asarray(table["P_det"],dtype=float)>=P_MIN
    n_surv = int(survivors.sum())
    print(f"{n_surv} of {len(table)} objects have P_det >= {P_MIN}"
          f"(V_lim = {V_LIM}, sigma_V = {SIGMA_V})")
    print(f"Expected number of detections: {np.nansum(table["P_det"]):.2f}")

    # table.sort("H",) # sort by H asc
    table.sort("P_det", reverse=True) # sort by P_det desc


    # 5. Plots
    plot_field(
        info,
        table, 
        angle_north,
        angle_east, 
        magnify=MAGNIFY,
        show = False, 
        title =f"All objects overlapping the image (ellipses x{MAGNIFY})",
        output_dir = OUTPUT_DIR_plots,
        output_file=f"{prefix}_candidates.png" if SAVE_FIGURES else None,
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
        output_dir= OUTPUT_DIR_plots,
        output_file=f"{prefix}_survivors.png" if SAVE_FIGURES else None,    
        )
    # plt.show()

    # 6. Table
    csv_file = f"{prefix}_results.csv"
    save_table_csv(table, OUTPUT_DIR_tables, csv_file, OUTPUT_COLUMNS)
    print(f"\nTable saved to {csv_file}")
    if PRINT_TABLE:
        print("\n--- Complete target results ---")
        table[OUTPUT_COLUMNS].pprint_all()

def main():

    if args.file != "Null":
        fits_files = [FITS_FILE]
    else:
        fits_files = sorted(FITS_DIR.glob("*.fits"))

    if not fits_files:
        print(f"No FITS files found in {FITS_DIR}")
        return

    for fits_file in fits_files:
        print(f"\nProcessing {fits_file}...")
        pipeline(fits_file)
        print("")


if __name__ == "__main__":
    main()
