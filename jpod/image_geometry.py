"""Image loading and geometry: WCS, orientation, ellipse angles, overlap filter."""

from dataclasses import dataclass

import astropy.units as u
from astropy.table import Column
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from .fits_header.fits_header import get_obs_times, get_observatory_code
from .fits_header.obscodes import load_obscodes_safe

@dataclass
class ImageInfo:
    """Everything we need to know about the FITS image."""
    data: np.ndarray
    wcs: WCS
    nx: int
    ny: int
    center: SkyCoord          # sky position of the image center
    corners: SkyCoord         # sky positions of the 4 corners
    pixel_scale: float        # arcsec/pixel (square pixels assumed)
    observatory_code: str     # MPC observatory code
    obs_code_source: str      # where the code was found
    date_obs: Time            # time written in the header
    start_obs: Time           # start of the exposure (UTC)
    end_obs: Time             # end of the exposure (UTC)
    exptime: object           # exposure time (Quantity) or None
    time_source: str          # where the time was found


def load_image(fits_file: str, observatory_code = None, time_reference = None, date_obs = None, obscodes_json = None)-> ImageInfo:
    """Read the FITS file and return an ImageInfo.
    observatory_code : fallback code used only if neither the header nor
        obscodes_json provides a code; otherwise it defaults to geocentric 500
    time_reference : None (automatic) or 'start', 'mid', 'end' to force what the time in the header means
    obscodes_json: path of the MPC JSON list of observatory codes
    date_obs: force the observation time (UTC string); if None it is taken from the header
    """
    with fits.open(fits_file) as hdul:
        header = hdul[0].header
        data = hdul[0].data.copy()

    nx, ny = header["NAXIS1"], header["NAXIS2"]
    w = WCS(header)

    center = SkyCoord.from_pixel((nx - 1) / 2, (ny - 1) / 2, w)
    corners = SkyCoord.from_pixel(
        np.array([0, nx - 1, nx - 1, 0]),
        np.array([0, 0, ny - 1, ny - 1]),
        w,
    )
    obscodes = load_obscodes_safe(obscodes_json)
    code, code_source = get_observatory_code(header, observatory_code, obscodes)
    date_obs, start_obs, end_obs, exptime, time_source = get_obs_times(header, time_reference, date_obs)

    scale = proj_plane_pixel_scales(w.celestial)[0] * 3600

    return ImageInfo(data, w, nx, ny, center, corners, scale,
                     code, code_source, date_obs, start_obs, end_obs, exptime, time_source)

def describe_time(info: ImageInfo)->list:
    """Lines describing the observation time, for the terminal."""
    lines = [f"Time in header: {info.date_obs.isot} UTC ({info.time_source})"]
    if info.exptime is not None:
        lines.append(f"Exposure: {info.exptime:.1f}, from {info.start_obs.isot} to {info.end_obs.isot} UTC.")
    return lines


def search_radius(info: ImageInfo, margin=0 * u.arcmin):
    """Radius of the circle containing the whole image, plus a margin."""
    distances = info.center.separation(info.corners)
    return np.max(distances).deg * u.deg + margin


def sky_direction_angle(w: WCS, coords, position_angle, step=10 * u.arcsec):
    """Angle in the image of a sky direction, at one or many positions.

    position_angle is measured from North through East (0 = N, 90 = E).
    The result is in degrees, counterclockwise from the +x axis
    (0 = right, 90 = up), assuming y increases upward in the image.
    """
    x0, y0 = w.world_to_pixel(coords)                             # starting pixel
    shifted = coords.directional_offset_by(position_angle, step)  # small step on the sky
    x1, y1 = w.world_to_pixel(shifted)                            # back to pixels
    return np.degrees(np.arctan2(y1 - y0, x1 - x0)) % 360.0


def image_orientation(w: WCS, coords):
    """Angle of North and East in the image, and the parity of the image.

    parity = +1 if East is counterclockwise from North, -1 if mirrored.
    """
    angle_north = sky_direction_angle(w, coords, 0 * u.deg)
    angle_east = sky_direction_angle(w, coords, 90 * u.deg)
    parity = 1 if ((angle_east - angle_north) % 360.0) < 180.0 else -1
    return angle_north, angle_east, parity


def add_ellipse_image_angle(table, w: WCS):
    """Add 'ellipse_angle': orientation of the uncertainty ellipse's major
    axis in the image (deg, CCW from +x, 0 = right, 90 = up), in [0, 180)."""
    angles = np.full(len(table), np.nan)

    pa = u.Quantity(table["LOV_PA"]).to(u.deg)
    ok = np.isfinite(pa.value)
    if ok.any():
        coords = SkyCoord(
            u.Quantity(table["RA"]).to(u.deg)[ok],
            u.Quantity(table["DEC"]).to(u.deg)[ok],
            frame="icrs",
        )
        # An ellipse is symmetric under a 180 deg rotation.
        angles[ok] = sky_direction_angle(w, coords, pa[ok]) % 180.0

    table["ellipse_angle"] = angles * u.deg


def ellipse_overlaps_image(x, y, a, b, angle_deg, nx, ny, n_samples=720)-> bool:
    """True if the ellipse (center x,y; semi-axes a,b in pixels; major axis at
    angle_deg CCW from +x) overlaps the image rectangle."""
    xmin, xmax, ymin, ymax = -0.5, nx - 0.5, -0.5, ny - 0.5

    def inside(px, py):
        return (px >= xmin) & (px <= xmax) & (py >= ymin) & (py <= ymax)

    # 1. center inside the image
    if inside(x, y):
        return True

    th = np.radians(angle_deg)
    c, s = np.cos(th), np.sin(th)

    # 2. some point of the outline inside the image
    t = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    ex = x + a * np.cos(t) * c - b * np.sin(t) * s
    ey = y + a * np.cos(t) * s + b * np.sin(t) * c
    if np.any(inside(ex, ey)):
        return True

    # 3. ellipse containing the image: some corner inside the ellipse
    cx = np.array([xmin, xmax, xmax, xmin]) - x
    cy = np.array([ymin, ymin, ymax, ymax]) - y
    u_ = cx * c + cy * s
    v_ = -cx * s + cy * c
    return bool(np.any((u_ / a) ** 2 + (v_ / b) ** 2 <= 1))


def filter_in_image(table, info: ImageInfo):
    """Keep objects whose position or uncertainty ellipse overlaps the image.
    Adds the columns 'x_pix' and 'y_pix' (pixel position of each object at the start of the exposure)
    and 'x_pix_end', 'y_pix_end'. Without the columns RA_end, DEC_end the two positions are the same.
    """
    nx, ny, scale = info.nx, info.ny, info.pixel_scale
    def to_pixel(ra_name, dec_name):
        coords = SkyCoord(u.Quantity(table[ra_name]).to(u.deg),
                          u.Quantity(table[dec_name]).to(u.deg), frame = "icrs")
        return info.wcs.world_to_pixel(coords)
    
    x, y = to_pixel("RA", "DEC")
    if "RA_end" in table.colnames:
        x_end, y_end = to_pixel("RA_end", "DEC_end")
    else:
        x_end, y_end = x, y

    a = u.Quantity(table["SMAA_3sigma"]).to_value(u.arcsec) / scale
    b = u.Quantity(table["SMIA_3sigma"]).to_value(u.arcsec) / scale
    ang = u.Quantity(table["ellipse_angle"]).to_value(u.deg)
    p = u.Quantity(table["posunc"]).to_value(u.arcsec) / scale

    def overlaps(i, xi, yi):
        if np.all(np.isfinite([a[i], b[i], ang[i]])):
            return ellipse_overlaps_image(xi, yi, a[i], b[i], ang[i], nx, ny)
        # Without a Horizons ellipse, use the provider's positional uncertainty.
        r = max(p[i] if np.isfinite(p[i]) else 0.0, 1e-6)
        return ellipse_overlaps_image(xi, yi, r, r, 0.0, nx, ny)

    keep = np.array([overlaps(i,x[i],y[i]) or overlaps(i, x_end[i], y_end[i])
                    for i in range(len(table))], dtype=bool)
    
    table["x_pix"] = x
    table["y_pix"] = y
    table["x_pix_end"] = x_end
    table["y_pix_end"] = y_end
    return table[keep]

def compute_trail_length(info: ImageInfo, table):
    """Compute the length (in pixels and arcseconds) of the trail produced by an object during the exposure. 
    It can be computed in two ways: 
    1) by using the starting and ending pixels, computing their distance and converting it to arcseconds using pixel_scale
    2) by multiplying sky_speed * exp_time and converting it to arcseconds and then to pixels. 
    We prefer the second way. 
    """
    if info.exptime is None:
        table["trail_length"] = np.full(len(table), np.nan) * u.arcsec
        _set_pixel_trail_column(table, np.full(len(table), np.nan))
        return

    sky_speed = table["sky_speed"] # already in arcsec/hour
    exptime = info.exptime.to(u.s) 
    pixel_scale = info.pixel_scale * u.arcsec / u.pixel
    trail_length = (sky_speed * exptime).to(u.arcsec)
    table["trail_length"] = trail_length
    pixel_trail = (trail_length / pixel_scale).to_value(u.pixel)
    _set_pixel_trail_column(table, pixel_trail)


def _set_pixel_trail_column(table, values):
    """Set the plain numeric pixel column with an explicit Astropy name."""
    column = Column(np.asarray(values), name="trail_length [px]")
    if "trail_length [px]" in table.colnames:
        table.replace_column("trail_length [px]", column)
    else:
        table.add_column(column)
