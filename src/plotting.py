"""Plot the field with objects, uncertainty ellipses and N/E arrows."""

import os

import astropy.units as u
from astropy.visualization import AsinhStretch, ImageNormalize, ZScaleInterval
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse


def plot_field(info, table, angle_north, angle_east, magnify=1,
               survivors = None, title = None, output_dir = None, 
               output_file = None, show = True):
    """Show the image on a WCS axis. 'table' must have the x_pix/y_pix columns
    (added by filter_in_image). Ellipses are magnified by 'magnify'.

    survivors: optional boolean array (one entry per row of table).
        If None, every object is drawn the same way (red fdot, orange ellipse).
        If given, objects with True are highlighted in green with their P_det and the others are drawn in grey with dashed ellipses
    
    """
    nx, ny = info.nx, info.ny
    norm = ImageNormalize(info.data, interval=ZScaleInterval(), stretch=AsinhStretch())

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection=info.wcs.celestial)
    ax.imshow(info.data, origin="lower", cmap="gray_r", norm=norm)
    ax.grid(color="cyan", ls=":", alpha=0.6)
    ax.set_xlabel("RA")
    ax.set_ylabel("Dec")

    for i, row in enumerate(table):
        x, y = float(row["x_pix"]), float(row["y_pix"])
        name = str(row["Name"]).strip()

        if survivors is None:
            color_pt, color_ell, ls, lw, text = "red", "orange", "-", 1.2, name
        elif survivors[i]:
            color_pt, color_ell, ls, lw = "tab:green", "tab:green", "-", 1.8
            text = f"{name} P={float(row["P_det"]):.2g}"
        else:
            color_pt, color_ell, ls, lw = "gray", "gray", "--", 1.0
            text = f"{name} P={float(row["P_det"]):.2g}"

        ax.plot(x, y, color=color_pt, ms=3)
        ax.text(x + 8, y + 8, text, color=color_pt, fontsize=8)

        a = u.Quantity(row["SMAA_3sigma"]).to_value(u.arcsec)
        b = u.Quantity(row["SMIA_3sigma"]).to_value(u.arcsec)
        ang = u.Quantity(row["ellipse_angle"]).to_value(u.deg)
        if np.all(np.isfinite([a, b, ang])):
            ax.add_patch(Ellipse(
                (x, y),
                width=2 * a * magnify / info.pixel_scale,   # full major axis, pixels
                height=2 * b * magnify / info.pixel_scale,  # full minor axis, pixels
                angle=ang,                                  # CCW from +x
                fill=False, ec=color_ell, lw=lw,
            ))

    # N/E arrows in the lower-left corner, from the angles at the image center
    length = 0.07 * min(nx,ny)
    x0, y0 = 2.0 * length, 2.0 * length
    for angle, label in ((angle_north, "N"), (angle_east, "E")):
        dx = length * np.cos(np.radians(angle))
        dy = length * np.sin(np.radians(angle))
        ax.annotate("", xy=(x0 + dx, y0 + dy), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color="yellow", lw=2))
        ax.text(x0 + 1.25 * dx, y0 + 1.25 * dy, label, color="yellow",
                ha="center", va="center", fontsize=12, weight="bold")

    ax.set_title(f"Uncertainty ellipses magnified x{magnify}")
    if output_file is not None:
        if output_dir is not None:
            output_file = os.path.join(output_dir, output_file)
        fig.savefig(output_file, dpi = 300, bbox_inches = "tight")
    if show:
        plt.show()
    return fig
