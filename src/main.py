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

MARGIN = 5 * u.arcmin    # widen the cone so ellipses of nearby objects are not missed
MAGNIFY = 10             # ellipse magnification in the plot (1 = true size)

OUTPUT_COLUMNS = [
    "Name", "RA", "DEC", "posunc", "V", "H",
    "RA_rate", "DEC_rate", "sky_speed",
    "SMAA_3sigma", "SMIA_3sigma", "LOV_PA", "ellipse_angle",
    "P_pos", "P_mag", "P_det",
]


def create_parser() -> argparse.ArgumentParser:
    """Parsing CLI arguments."""

    project_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", default=project_root / "data")  # FITS directory
    parser.add_argument("-f", "--file", default="Null")  # FITS file name
    parser.add_argument("-outdir", "--output_dir", default=project_root / "output")  # output directory

    parser.add_argument("-o", "--observatory", default="I41")  # observatory code
    parser.add_argument("-vlim", "--vlim", type=float, default=21.5)
    parser.add_argument("-sigv", "--sigma_v", type=float, default=0.5)
    parser.add_argument("-pmin", "--pmin", type=float, default=0.05)

    parser.add_argument("-savefig", "--savefig", action="store_true", default=False)
    parser.add_argument("-printtable", "--printtable", action="store_true", default=False)

    return parser


def pipeline(fits_file: str, args: argparse.Namespace, output_dir: Path):

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
    table = query_skybot(field, radius, info.epoch, args.observatory)
    if len(table) == 0:
        print("No asteroids found in this field of view.")
        return

    # 3. JPL data, ellipse orientation in the image, overlap filter
    fetch_jpl_data(table, info.epoch, args.observatory)

    # Filter out objects with excessively large uncertainties (e.g. > 10 arcmin)
    max_uncertainty = 600 * u.arcsec 
    smaa = u.Quantity(table["SMAA_3sigma"])
    threshold = (smaa < max_uncertainty) | np.isnan(smaa.value)
    table = table[threshold]
    print(f"\nFiltered out {np.sum(~threshold)} objects with excessively large uncertainties.")

    add_ellipse_image_angle(table, info.wcs)
    table = filter_in_image(table, info)
    print(f"\n{len(table)} objects overlap the image.")
    if len(table) == 0:
        return

    # 4. Detection probability and survivors
    add_detection_probability(table, info, args.vlim, args.sigma_v)
    survivors = np.asarray(table["P_det"], dtype=float) >= args.pmin
    n_surv = int(survivors.sum())
    print(f"{n_surv} of {len(table)} objects have P_det >= {args.pmin}"
          f"(V_lim = {args.vlim}, sigma_V = {args.sigma_v})")
    print(f"Expected number of detections: {np.nansum(table['P_det']):.2f}")

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
        output_dir=output_dir / "plots",
        output_file=f"{prefix}_candidates.png" if args.savefig else None,
        )
    plot_field(
        info, 
        table, 
        angle_north, 
        angle_east, 
        magnify=MAGNIFY, 
        show = False, 
        survivors=survivors, 
        title = (f"Green: P_det >= {args.pmin} (V_lim ={args.vlim});"
                f"grey: below threshold (ellipses x{MAGNIFY})"),
        output_dir=output_dir / "plots",
        output_file=f"{prefix}_survivors.png" if args.savefig else None,
        )
    # plt.show()

    # 6. Table
    csv_file = f"{prefix}_results.csv"
    save_table_csv(table, output_dir / "tables", csv_file, OUTPUT_COLUMNS)
    print(f"\nTable saved to {csv_file}")
    if args.printtable:
        print("\n--- Complete target results ---")
        table[OUTPUT_COLUMNS].pprint_all()

def main():

    args = create_parser().parse_args()

    fits_dir = Path(args.dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True) # create is not exist
    (output_dir / "plots").mkdir(exist_ok=True) # create is not exist
    (output_dir / "tables").mkdir(exist_ok=True)

    if args.file != "Null":
        fits_files = [fits_dir / args.file]
    else:
        fits_files = sorted(fits_dir.glob("*.fits"))

    if not fits_files:
        print(f"No FITS files found in {fits_dir}")
        return

    for fits_file in fits_files:
        print(f"\nProcessing {fits_file}...")
        pipeline(fits_file, args, output_dir)
        print("")


if __name__ == "__main__":
    main()
