"""Plot the field with objects, uncertainty ellipses and N/E arrows."""

import os

import astropy.units as u
from astropy.visualization import AsinhStretch, ImageNormalize, ZScaleInterval
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse
from matplotlib.lines import Line2D


def _row_quantity_value(row, table, column, unit):
    value = row[column]
    if hasattr(value, "to_value"):
        return value.to_value(unit)
    return (value * table[column].unit).to_value(unit)


def plot_field(info, table, angle_north, angle_east, magnify=1, object_class = None, 
               survivors = None, title = None, output_dir = None, 
               output_file = None, show = True, view_margin = 0.05, V_lim =None, P_min=None):
    """Show the image on a WCS axis. 'table' must have the x_pix/y_pix columns
    (added by filter_in_image). Ellipses are magnified by 'magnify'.

    object_class: optional string to restrict the plotting only to objects of that class (e.g NEAs). 
        If the class is absent, all objects are plotted with a warning


    survivors: optional boolean array (one entry per row of table).
        If None, every object is drawn the same way (red fdot, orange ellipse).
        If given, objects with True are highlighted in green with their P_det and the others are drawn in grey with dashed ellipses
    
    view_margin: the view is the image plus a white margin of this fraction of the larger image size, to show objects near the borders
    """

   # Filter by object class, if requested
    if object_class is not None:
        mask = np.asarray(table["class"] == object_class, dtype=bool)
        if np.any(mask):
            table = table[mask]
            if survivors is not None:
                survivors = np.asarray(survivors)[mask]
        else:
            print(f"No objects of class {object_class} found in table. Plotting all objects instead.")
            object_class = None

    nx, ny = info.nx, info.ny
    norm = ImageNormalize(info.data, interval=ZScaleInterval(), stretch=AsinhStretch())

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection=info.wcs.celestial)
    ax.imshow(info.data, origin="lower", cmap="gray_r", norm=norm)
    ax.grid(color="cyan", ls=":", alpha=0.6)
    ax.set_xlabel("RA")
    ax.set_ylabel("Dec")

    biggest = (0.0, "", 0.0) # drawn semi-major axis in pixels, name, true one
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

        ax.plot(x, y, ".", color=color_pt, ms=3)
        ax.text(x + 8, y + 8, text, color=color_pt, fontsize=8)

        a = _row_quantity_value(row, table, "SMAA_3sigma", u.arcsec)
        b = _row_quantity_value(row, table, "SMIA_3sigma", u.arcsec)
        ang = _row_quantity_value(row, table, "ellipse_angle", u.deg)
        if np.all(np.isfinite([a, b, ang])):
            ax.add_patch(Ellipse(
                (x, y),
                width=2 * a * magnify / info.pixel_scale,   # full major axis, pixels
                height=2 * b * magnify / info.pixel_scale,  # full minor axis, pixels
                angle=ang,                                  # CCW from +x
                fill=False, ec=color_ell, ls=ls, lw=lw,
            ))
            drawn = a * magnify / info.pixel_scale
            if drawn > biggest[0]:
                biggest = (drawn, name, a/info.pixel_scale)

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

    if view_margin is not None:
        pad = view_margin * max(nx,ny)
        ax.set_xlim(-0.5 - pad, nx -0.5 + pad)
        ax.set_ylim(-0.5 - pad, ny - 0.5 + pad)
        if biggest[0] > max(nx,ny):
            print(f"Note: the ellipse of {biggest[1]} is huge ({biggest[2]:.0f} px semi-major axis, {biggest[0]:.0f} px at x{magnify}); only the part inside the view is shown")
    # Title
    if object_class is None:
        object_text = "All objects"
    else:
        object_text = f"All {object_class}s"
    if survivors is None:
        title_text = (
            f"{object_text} overlapping the image "
            f"(ellipses magnified x{magnify}).\n"
            "Positions at the start of the exposure"
        )
    else:
        title_text = (
            f"{object_text} overlapping the image. "
            f"Green: detectable (V_lim={V_lim}, P_min={P_min}); "
            "grey: not detectable.\n"
            f"Ellipses magnified x{magnify}. "
            "Positions at the start of the exposure"
        )
    ax.set_title(title or title_text)

    if output_file is not None:
        if output_dir is not None:
            output_file = os.path.join(output_dir, output_file)
        fig.savefig(output_file, dpi = 300, bbox_inches = "tight")
    if show:
        plt.show()
    return fig

def _number(value):
    """Plain float from a table value (Quantity, masked or plain number)."""
    return float(getattr(value, "value", value))
 
 
def _find_row(table, query):
    """Row whose name equals 'query', or else contains it (ignoring case)."""
    query = query.strip().lower()
    names = [str(n).strip().lower() for n in table["Name"]]
    for i, name in enumerate(names):
        if name == query:
            return table[i]
    for i, name in enumerate(names):
        if query in name:
            return table[i]
    return None
 
 
def plot_cutouts(info, table, names, 
                half_size=60, magnify=1, motion_scale=1,
                output_dir = None, output_file=None, show=True):
    """Zoom around the objects whose name matches one of 'names'.
 
    Each panel shows (in pixel coordinates, 2 * half_size pixels wide):
      red circle   predicted position (mid-exposure), drawn open so that it
                   does not hide a source
      blue line    expected motion from the start to the end of the exposure; need the columns x_pix_end, y_pix_end
      orange       uncertainty ellipse at the start (magnified by 'magnify')
      red dashed   edges of the image, when they are inside the panel
    """
    rows = []
    for query in names:
        row = _find_row(table, query)
        if row is None:
            print(f"Zoom: no object matching {query!r} in the table (it may be "
                  "outside the image, or filtered out).")
        else:
            rows.append(row)
    if not rows:
        return None
    has_end = "x_pix_end" in table.colnames and "y_pix_end" in table.colnames
    fig, axes = plt.subplots(1, len(rows), figsize=(6 * len(rows), 6.4), squeeze=False)
 
    for ax, row in zip(axes[0], rows):
        name = str(row["Name"]).strip()
        x, y = float(row["x_pix"]), float(row["y_pix"])
        x_end, y_end = (float(row["x_pix_end"]), float(row["y_pix_end"])) if has_end else (x,y)
        trail_length = float(row["trail_length [px]"])

        # center of the panel: middle of the motion
        cx, cy = (x+x_end)/2, (y+y_end)/2
        half = (
            max(half_size, int(np.ceil(0.6 * trail_length + 10)))
            if np.isfinite(trail_length)
            else half_size
        )
 
        # part of the image inside the panel (can be empty if far outside)
        xlo, xhi = max(0, int(np.floor(cx)) - half), min(info.nx, int(np.ceil(cx)) + half + 1)
        ylo, yhi = max(0, int(np.floor(cy)) - half), min(info.ny, int(np.ceil(cy)) + half + 1)
        if xhi > xlo and yhi > ylo:
            cut = info.data[ylo:yhi, xlo:xhi]
            norm = ImageNormalize(cut, interval=ZScaleInterval(), stretch=AsinhStretch())
            ax.imshow(cut, origin="lower", cmap="gray_r", norm=norm,
                      extent=(xlo - 0.5, xhi - 0.5, ylo - 0.5, yhi - 0.5))
 
        # edges of the image
        for edge in (-0.5, info.nx - 0.5):
            ax.axvline(edge, color="red", ls="--", lw=0.8)
        for edge in (-0.5, info.ny - 0.5):
            ax.axhline(edge, color="red", ls="--", lw=0.8)
 
        # predicted position at the start of the exposure
        ax.plot(
            x,y,
            marker="o",
            ms=5,
            mfc="none",
            mec="red",
            mew=1.3,
            ls="",
            zorder=5
        )
        handles = [
            Line2D(
                [],[], 
                color = "red",
                marker ="o",
                mfc="none", 
                ls="",
                label="position at the start"
            )
        ]

        # expected motion from the start to the end of the exposure
        if has_end:
            dx = x_end -x
            dy = y_end -y
            motion = np.hypot(dx,dy)
            if motion > 0:
                arrow_length = max(motion * motion_scale, 5.0)
                ux, uy = dx/motion, dy/motion
                ax.annotate(
                    "",
                    xy=(x+ux*arrow_length, y+uy*arrow_length),
                    xytext=(x,y),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color="tab:blue",
                        lw=2,
                        mutation_scale=13,
                        shrinkA=3,
                        shrinkB=0
                    ),
                    zorder=6
                )
            handles.append(
                Line2D(
                    [],[],
                    color="tab:blue", 
                    marker=">", 
                    ls="-",
                    lw=2,
                    label=f"Motion direction (x{motion_scale})"))

        # uncertainty ellipse (at the start)
        a = _row_quantity_value(row, table, "SMAA_3sigma", u.arcsec)
        b = _row_quantity_value(row, table, "SMIA_3sigma", u.arcsec)
        ang = _row_quantity_value(row, table, "ellipse_angle", u.deg)
        if np.all(np.isfinite([a, b, ang])):
            ax.add_patch(Ellipse((x, y), 2 * a * magnify / info.pixel_scale,
                                 2 * b * magnify / info.pixel_scale, angle=ang,
                                 fill=False, ec="orange", lw=1.2, alpha =1, zorder=4))
 
        info_text = f"V = {_number(row['V']):.1f}"
        if "P_det" in table.colnames:
            info_text += f"   P_det = {_number(row['P_det']):.2g}"
        motion_text = f"\nMotion during the exposure: {motion:.1f} px" if has_end else ""
        ax.set_title(f"{name}\n{info_text}   pixel ({x:.0f}, {y:.0f}){motion_text}\n Ellipses magnified x{magnify}, motion magnified x{motion_scale}",
                     fontsize=10)
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy - half, cy + half)
        ax.set_aspect("equal")
        ax.set_xlabel("x (pixels)")
        ax.set_ylabel("y (pixels)")
        ax.legend(handles=handles,loc="upper right", fontsize=8)
 
    fig.tight_layout()
    if output_file is not None:
        if output_dir is not None:
            output_file = os.path.join(output_dir, output_file)
        fig.savefig(output_file, dpi = 300, bbox_inches = "tight")
    if show:
        plt.show()
    return fig
