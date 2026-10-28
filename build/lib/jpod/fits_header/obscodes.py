"""Find an MPC observatory code in the MPC JSON list of codes.

The list is only used when the header has no observatory code. The site is
identified by its coordinates (if the header has them, like OBSLON, OBSLAT) and, if that fails, by
its name: in this case it compares the significant words of TELESCOP and OBSERVAT with the "Name" field in the json file.
A code is returned only when the match is unambiguous:

- if several codes are at the same place (e.g. several programmes at the same
  mountain), any of them gives the same observer position, so the one with the
  most generic name is used and the others are listed;
- if the candidates are at different places, nothing is guessed. In this case you get a warning and the geocentric code 500,
  or you can set OBSERVATORY_CODE by hand.
"""

import json
import math
import re
from functools import lru_cache

# Keys used by the JSON file.
KEY_LONGITUDE, KEY_COS, KEY_SIN, KEY_NAME = "Longitude", "cos", "sin", "Name"

# Header keywords with the site position: degrees (east positive), degrees
# (north positive) and metres above sea level. Only numeric values are used.
LON_KEYWORDS = ("OBSLON", "LONGITUD")
LAT_KEYWORDS = ("OBSLAT", "LATITUDE")
ALT_KEYWORDS = ("OBSALT", "ALTITUDE")

# Header keywords with the telescope / site name.
SITE_NAME_KEYWORDS = ("TELESCOP", "OBSERVAT")

# Words that say nothing about the site.
GENERIC_WORDS = {"observatory", "observatorio", "observatoire", "telescope",
                 "telescopio", "station", "obs", "inch", "the", "and"}

# Distances are in Earth radii (1 Earth radius = 6378 km).
SAME_SITE = 0.002      # codes closer than this (~13 km) count as the same site
MAX_DISTANCE = 0.005   # max distance (~32 km) between header and list position

WGS84_F = 1.0 / 298.257223563


def _xyz(lon_deg, rho_cos, rho_sin):
    lon = math.radians(lon_deg)
    return (rho_cos * math.cos(lon), rho_cos * math.sin(lon), rho_sin)


def site_xyz(lon_deg, lat_deg, alt_m=0.0):
    """Position of a site in Earth radii, from geodetic longitude, latitude
    and altitude. Uses the same parallax constants as the MPC list."""
    a = 6378137.0
    lat = math.radians(lat_deg)
    c = 1.0 / math.sqrt(math.cos(lat) ** 2 + ((1 - WGS84_F) * math.sin(lat)) ** 2)
    s = (1 - WGS84_F) ** 2 * c
    rho_cos = (c + alt_m / a) * math.cos(lat)
    rho_sin = (s + alt_m / a) * math.sin(lat)
    return _xyz(lon_deg, rho_cos, rho_sin)


def _tokens(text):
    """Significant lowercase words (letters only: aperture sizes are dropped)."""
    return {w for w in re.findall(r"[a-z]{3,}", str(text).lower())
            if w not in GENERIC_WORDS}


@lru_cache(maxsize=None)
def load_obscodes(path):
    """Read the JSON list: {code: {position, name, name words}}. Entries
    without a fixed position (satellites, roving observers) are skipped."""
    with open(path, encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("the JSON file is not a dictionary of observatory codes")

    codes = {}
    for code, entry in raw.items():
        try:
            lon = float(entry[KEY_LONGITUDE])
            rho_cos = float(entry[KEY_COS])
            rho_sin = float(entry[KEY_SIN])
        except (KeyError, TypeError, ValueError):
            continue
        if rho_cos == 0 and rho_sin == 0:
            continue
        name = str(entry.get(KEY_NAME, "")).strip()
        codes[str(code)] = {"xyz": _xyz(lon, rho_cos, rho_sin),
                            "name": name, "tokens": _tokens(name)}
    return codes


def load_obscodes_safe(path):
    """Like load_obscodes, but returns None (with a warning) on failure."""
    if not path:
        return None
    try:
        return load_obscodes(str(path))
    except (OSError, ValueError) as exc:
        print(f"WARNING: could not read the observatory code list {path!r} ({exc}).")
        return None


def _first_number(header, keywords):
    for key in keywords:
        if key in header:
            try:
                value = float(header[key])
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                return value
    return None


def _by_coordinates(obscodes, xyz):
    """Codes at the position xyz: the nearest one and those at its site."""
    near = sorted((math.dist(xyz, e["xyz"]), code) for code, e in obscodes.items())
    near = [(d, code) for d, code in near if d <= MAX_DISTANCE]
    if not near:
        return []
    first = obscodes[near[0][1]]["xyz"]
    return [code for _, code in near
            if math.dist(first, obscodes[code]["xyz"]) <= SAME_SITE]


def _by_name(obscodes, names):
    """Codes whose name contains all the significant words of a header name."""
    wanted = [t for t in (_tokens(n) for n in names) if t]
    return [code for code, e in obscodes.items()
            if any(t <= e["tokens"] for t in wanted)]


def _same_site(obscodes, group):
    first = obscodes[group[0]]["xyz"]
    return all(math.dist(first, obscodes[c]["xyz"]) <= SAME_SITE for c in group)


def _describe(obscodes, group, header_tokens, how):
    best = min(group, key=lambda c: (-len(header_tokens & obscodes[c]["tokens"]),
                                     len(obscodes[c]["name"]), c))
    others = [c for c in group if c != best]
    text = f"MPC list: {best} ({obscodes[best]['name']}), matched by {how}"
    if others:
        shown = ", ".join(others[:5]) + (", ..." if len(others) > 5 else "")
        text += f"; {len(others)} other code(s) at the same site: {shown}"
    return best, text


def find_code(obscodes, header):
    """Return (code, message). code is None if no unambiguous match exists."""
    names = [str(header[k]).strip() for k in SITE_NAME_KEYWORDS if k in header]
    header_tokens = set()
    for name in names:
        header_tokens |= _tokens(name)

    lon = _first_number(header, LON_KEYWORDS)
    lat = _first_number(header, LAT_KEYWORDS)
    if lon is not None and lat is not None and -90 <= lat <= 90:
        alt = _first_number(header, ALT_KEYWORDS) or 0.0
        group = _by_coordinates(obscodes, site_xyz(lon, lat, alt))
        if group:
            return _describe(obscodes, group, header_tokens, "coordinates")

    group = _by_name(obscodes, names)
    if group:
        if _same_site(obscodes, group):
            return _describe(obscodes, group, header_tokens, "name")
        listing = ", ".join(f"{c} ({obscodes[c]['name']})" for c in group[:5])
        return None, (f"the name {names} matches codes at different places "
                      f"({listing}{', ...' if len(group) > 5 else ''})")

    return None, "no match by coordinates or by name"