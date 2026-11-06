"""Find asteroids in FITS images with a selected provider and enrich via Horizons."""
import argparse
from pathlib import Path

import numpy as np

from . import __version__
from .horizons import enrich_with_horizons, set_cache_db
from .image_geometry import (
    add_ellipse_image_angle,
    filter_in_image,
    image_orientation,
    load_image,
    describe_time,
    search_radius,
    compute_trail_length,
)
from .providers import query_mpc_identification, query_skybot
from .plotting import plot_field, plot_cutouts
from .detection import add_detection_probability
from .trailing_loss import add_trailing_loss
from .output import save_table_csv
from .config import (
    MARGIN,
    OBSCODES_JSON,
    OUTPUT_COLUMNS,
    ZOOM_HALF_SIZE,
    resolve_v50_catalog_dir,
)
from .photometric_calibration import (
    calculate_limiting_magnitude,
    determine_zero_point,
    is_v_filter,
)
from .image_analysis import measure_image

MOTION_SCALE = 10
ZOOM_OBJECTS = ["2020 UL3"]


DATE_OBS = None         # None: read it from the header, or force it
TIME_REFERENCE = None   # what DATE-OBS (or similar keywords) means: "start", "mid" or "end" of the exposure


def resolve_path(path, workdir, default_subdir=None):
    """
    Resolves path:
    - If path is absolute, it is used as-is.
    - If path is relative, it is joined with the working directory.
    - If path is None, it defaults to workdir / default_subdir.
    """
    if path is None:
        return (workdir / default_subdir).resolve() if default_subdir else workdir.resolve()
    
    p = Path(path)
    if p.is_absolute():
        return p.resolve()
    else:
        return (workdir / p).resolve()


def _ensure_pixel_trail_length(info, table):
    """Restore the pixel trail column if a downstream step removed it."""
    if "trail_length [px]" not in table.colnames:
        print(
            "Warning: pixel trail length is missing; recalculating trail lengths "
            "before saving."
        )
        compute_trail_length(info, table)


def _positive_float(value: str) -> float:
    """Parse a finite positive CLI value such as the SNR threshold."""
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not np.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return parsed


def create_parser() -> argparse.ArgumentParser:
    """Parsing CLI arguments."""

    # project_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("-w", "--workdir", type=Path, default=None, help="Working directory. Defaults to the current folder")

    parser.add_argument("-d", "--dir", type=Path, default=None, help="FITS files directory")
    parser.add_argument("-f", "--file", type=Path, default=None, help="FITS file name with extension. Use if you want to process a single file.\n" 
                        +"Default parent directory is the data directory")
    parser.add_argument("--output_dir", type=Path, default=None, help="Output directory")

    parser.add_argument("-obs", "--observatory", type=str, default=None,
                        help="Fallback observatory code when header/site lookup finds none")
    parser.add_argument(
        "--provider",
        choices=("mpc", "skybot"),
        default="skybot",
        help="Small-body discovery provider (default: skybot)",
    )
    parser.add_argument(
        "--vlim",
        type=float,
        default=21.5,
        help="Fallback limiting magnitude if an automatic V-band estimate is unavailable",
    )
    parser.add_argument(
        "--snr",
        type=_positive_float,
        default=3.0,
        help="SNR threshold for the automatic sky-noise limiting magnitude (default: 3)",
    )
    parser.add_argument("--sigmav", type=float, default=0.5, help="Standard deviation of the magnitude")
    parser.add_argument("--pmin", type=float, default=0.05, help="Minimum detection probability")
    parser.add_argument("--magnify", type=float, default=1.0, help="Magnification factor for the ellipses in the plots")

    parser.add_argument("--no-cache", action="store_true", default=False, help="Disable local cache and force fresh API queries.")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Relative path of the JPL cache DB.\n" 
                        + "Specify the file name too: the extension must be .db.\n"
                        + "Defaults to <workdir>/database/jpl_horizons_cache.db")

    parser.add_argument("--savefig", action="store_true", default=False, help="Save the plots as PNG files")
    parser.add_argument("--printtable", action="store_true", default=False, help="Print the complete table of results")

    parser.add_argument(
        "-V", "--version", 
        action="version", 
        version=f"%(prog)s {__version__}", help="Show the version of the program and exit"
    )

    return parser

def pipeline(fits_file: str, args: argparse.Namespace, output_dir: Path):

    prefix = Path(fits_file).stem
    # 1. Load the FITS image and resolve its WCS, observatory code, and times
    #    before making any provider requests.
    info = load_image(
        fits_file, args.observatory, TIME_REFERENCE, DATE_OBS, OBSCODES_JSON
    )

    # 2. Report the resolved image centre, observation interval, and orientation.
    print(f"\nObservatory code: {info.observatory_code} ({info.obs_code_source})\n")
    print()
    for line in describe_time(info):
        print(line)
    print()
    angle_north, angle_east, parity = image_orientation(info.wcs, info.center)
    # Background, FWHM, detected stars, and aperture fluxes share one analysis.
    measurements = measure_image(info)
    fwhm_pixels = measurements.fwhm_pixels
    print(f"Stellar FWHM: {fwhm_pixels:.2f} pixels")
    workdir = args.workdir.resolve() if args.workdir else Path.cwd()
    catalog_dir = resolve_v50_catalog_dir(workdir)
    zero_point = determine_zero_point(info, measurements, catalog_dir)
    v_lim = args.vlim
    if zero_point.value is not None:
        if zero_point.matched_stars:
            print(
                f"Photometric zero point: {zero_point.value:.3f} "
                f"({zero_point.source}; {zero_point.matched_stars} stars, "
                f"robust scatter {zero_point.scatter:.3f} mag)",
                flush=True,
            )
        else:
            print(
                f"Photometric zero point: {zero_point.value:.3f} "
                f"({zero_point.source})",
                flush=True,
            )
        if is_v_filter(info.filter_name):
            limiting_magnitude = calculate_limiting_magnitude(
                info,
                measurements,
                zero_point.value,
                args.snr,
            )
            if limiting_magnitude.value is not None:
                v_lim = limiting_magnitude.value
                gain_note = (
                    f", gain {limiting_magnitude.gain_e_per_adu:.3g} "
                    "electrons/image-unit"
                    if limiting_magnitude.gain_e_per_adu is not None
                    else ", assuming 1 image unit per electron (no gain in header)"
                )
                print(
                    f"Median sky background: "
                    f"{limiting_magnitude.background_rate:.3f} image units/"
                    f"pixel/second; sky noise in the FWHM aperture: "
                    f"{limiting_magnitude.sky_noise:.3f} image units"
                    f"{gain_note}.",
                    flush=True,
                )
                print(
                    f"Calculated V_lim: {v_lim:.2f} "
                    f"(SNR threshold {args.snr:g}); using this value.",
                    flush=True,
                )
            else:
                print(
                    f"Automatic V_lim unavailable: "
                    f"{limiting_magnitude.unavailable_reason}. "
                    f"Continuing with --vlim={v_lim:.2f}.",
                    flush=True,
                )
        else:
            print(
                f"Automatic V_lim is currently supported only for V-band "
                f"images. Continuing with --vlim={v_lim:.2f}.",
                flush=True,
            )
    else:
        print(
            f"Photometric zero point unavailable: {zero_point.unavailable_reason}. "
            f"Continuing with --vlim={args.vlim:.2f}.",
            flush=True,
        )
    print()
    print(
        f"At the image center: North points at {angle_north:.4f} deg, "
        f"East at {angle_east:.4f} deg (CCW from +x, 0 = right, 90 = up); "
        f"parity = {parity:+d}"
    )
    print()

    # 3. Discover small bodies; both adapters return the shared provider schema.
    radius = search_radius(info, MARGIN)
    if args.provider == "mpc":
        print(
            f"Querying MPC MPChecker for minor planets in the image field "
            f"in a radius of {radius}...",
            flush=True,
        )
        table = query_mpc_identification(info, radius)
    else:
        print(
            f"Querying SkyBoT for small bodies in the image field "
            f"in a radius of {radius}...",
            flush=True,
        )
        table = query_skybot(
            info.center, radius, info.start_obs, info.observatory_code
        )
    if len(table) == 0:
        print("No small bodies found in this field of view.")
        return

    # 4. Enrich candidates with Horizons data, then keep the image-overlapping
    #    objects and measure their projected trail lengths.
    table = enrich_with_horizons(
        table,
        info.start_obs, info.end_obs, info.observatory_code,
        use_cache = not args.no_cache
    )
    add_ellipse_image_angle(table, info.wcs)
    table = filter_in_image(table, info)
    compute_trail_length(info, table)
    add_trailing_loss(table, fwhm_pixels)
    print(f"\n{len(table)} objects overlap the image.")
    if len(table) == 0:
        return

    # 5. Combine positional and magnitude probabilities into the final score.
    add_detection_probability(table, info, v_lim, args.sigmav)
    survivors = np.asarray(table["P_det"], dtype=float) >= args.pmin
    table["detectable"] = survivors
    n_surv = int(survivors.sum())
    print(f"{n_surv} of {len(table)} objects have P_det >= {args.pmin}"
          f"(V_lim = {v_lim}, sigma_V = {args.sigmav})")
    print(f"Expected number of detections: {np.nansum(table['P_det']):.2f}")
    table.sort("P_det", reverse=True) # sort by P_det desc
    survivors = np.asarray(table["detectable"], dtype=bool) #rebuilt after sorting


    # 6. Render candidate and detection-probability plots.
    plot_field(
        info,
        table, 
        angle_north,
        angle_east, 
        magnify=args.magnify,
        object_class="NEA",
        survivors=None,
        show=False,
        title = None,
        output_dir=output_dir / "plots",
        output_file=f"{prefix}_candidates.png" if args.savefig else None,
        V_lim=v_lim,
        P_min=args.pmin
        )
    plot_field(
        info, 
        table, 
        angle_north, 
        angle_east, 
        magnify=args.magnify, 
        object_class=None,
        survivors=survivors, 
        show=False,
        title = None,
        output_dir=output_dir / "plots",
        output_file=f"{prefix}_survivors.png" if args.savefig else None,
        V_lim=v_lim,
        P_min=args.pmin
        )
    if ZOOM_OBJECTS:
        plot_cutouts(
            info, 
            table, 
            ZOOM_OBJECTS, 
            ZOOM_HALF_SIZE, 
            magnify=args.magnify, 
            motion_scale = MOTION_SCALE, 
            output_dir = output_dir / "plots",
            output_file=f"{prefix}_zoom.png" if args.savefig else None,
            show=False 
            )   
    #plt.show()

    # 7. Save only the compact, user-facing result columns to CSV.
    _ensure_pixel_trail_length(info, table)
    csv_file = f"{prefix}_results.csv"
    save_table_csv(table, output_dir / "tables", csv_file, OUTPUT_COLUMNS)
    print(f"\nTable saved to {output_dir}/tables/{csv_file}")
    if args.printtable:
        print("\n--- Complete target results ---")
        table[OUTPUT_COLUMNS].pprint_all()

def main():

    args = create_parser().parse_args()

    workdir = args.workdir.resolve() if args.workdir else Path.cwd()

    if args.cache_dir is not None:
        cache_db = resolve_path(args.cache_dir, workdir)
    else:
        (workdir / "database").mkdir(exist_ok=True) # create if not exist
        cache_db = workdir / "database/jpl_horizons_cache.db"
    set_cache_db(cache_db)

    fits_dir = resolve_path(args.dir, workdir, default_subdir="data")
    output_dir = resolve_path(args.output_dir, workdir, default_subdir="output")

    output_dir.mkdir(parents=True, exist_ok=True) # create if not exist
    (output_dir / "plots").mkdir(exist_ok=True) # create if not exist
    (output_dir / "tables").mkdir(exist_ok=True)

    fits_files = [fits_dir / args.file] if args.file else sorted(fits_dir.glob("*.fits"))

    if not fits_files:
        print(f"No FITS files found in {fits_dir}")
        return

    for fits_file in fits_files:
        print(f"\nProcessing {fits_file}...")
        pipeline(fits_file, args, output_dir)
        print("")

if __name__ == "__main__":
    main()
