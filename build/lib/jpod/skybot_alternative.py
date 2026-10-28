"""MPC MPChecker discovery adapted to the existing catalog pipeline."""

import re
from datetime import timedelta
from html.parser import HTMLParser

import astropy.units as u
import numpy as np
import requests
from astropy.coordinates import Angle
from astropy.table import Table

from image_geometry import ImageInfo


MPC_CHECKER_FORM_URL = "https://minorplanetcenter.net/cgi-bin/checkmp.cgi"
MPC_CHECKER_URL = "https://minorplanetcenter.net/cgi-bin/mpcheck.cgi"
MPC_CHECKER_MAG_LIMIT = 30.0
MPC_CHECKER_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)


class _PreformattedText(HTMLParser):
    """Collect MPChecker's fixed-width results from its preformatted block."""

    def __init__(self):
        super().__init__()
        self._in_pre = False
        self.parts = []

    def handle_starttag(self, tag, attrs):
        del attrs
        if tag == "pre":
            self._in_pre = True

    def handle_endtag(self, tag):
        if tag == "pre":
            self._in_pre = False

    def handle_data(self, data):
        if self._in_pre:
            self.parts.append(data)


_MPC_RESULT_ROW = re.compile(
    r"^\s*(?P<number>\(\d+\)\s*)?"
    r"(?P<name>.+?)\s{2,}"
    r"(?P<ra_h>\d{1,2})\s+(?P<ra_m>\d{1,2})\s+(?P<ra_s>\d+(?:\.\d+)?)\s+"
    r"(?P<dec_d>[+-]\d{1,2})\s+(?P<dec_m>\d{1,2})\s+(?P<dec_s>\d+(?:\.\d+)?)\s+"
    r"(?P<mag>\d+(?:\.\d+)?)\s+"
    r"\S+\s+\S+\s+"
    r"(?P<ra_rate>\d+(?:\.\d+)?[+-])\s+"
    r"(?P<dec_rate>\d+(?:\.\d+)?[+-])\s+"
    r"\d+\s*o?\b"
)


def _signed_rate(value):
    """Parse MPChecker's arcseconds/hour rate, e.g. ``3-``."""
    return float(value[:-1]) * (-1 if value.endswith("-") else 1)


def _mpchecker_table(html):
    if "Error from WebCS Script" in html or "Incorrect Form Entry" in html:
        message = re.sub(r"<[^>]+>", " ", html)
        message = " ".join(message.split())
        raise ValueError(f"MPC MPChecker rejected the query: {message}")

    parser = _PreformattedText()
    parser.feed(html)
    if not parser.parts:
        if "No known minor planets" in html:
            return _empty_mpc_table()
        raise ValueError("MPC MPChecker response did not contain a results block")

    names = []
    numbers = []
    ra_values = []
    dec_values = []
    ra_rates = []
    dec_rates = []
    magnitudes = []
    for line in "".join(parser.parts).splitlines():
        match = _MPC_RESULT_ROW.match(line)
        if match is None:
            continue

        row = match.groupdict()
        name = row["name"].strip()
        if row["number"]:
            name = f'{row["number"].strip()} {name}'
        ra = Angle(
            f'{row["ra_h"]}h{row["ra_m"]}m{row["ra_s"]}s'
        ).to_value(u.deg)
        dec_sign = -1 if row["dec_d"].startswith("-") else 1
        dec = dec_sign * (
            abs(int(row["dec_d"]))
            + int(row["dec_m"]) / 60
            + float(row["dec_s"]) / 3600
        )

        names.append(name)
        numbers.append(_number_from_name(name))
        ra_values.append(ra)
        dec_values.append(dec)
        ra_rates.append(_signed_rate(row["ra_rate"]) / 60)
        dec_rates.append(_signed_rate(row["dec_rate"]) / 60)
        magnitudes.append(float(row["mag"]))

    if not names and "No known minor planets" not in html:
        raise ValueError("MPC MPChecker returned results, but no object rows could be parsed")
    return _make_mpc_table(names, numbers, ra_values, dec_values, ra_rates, dec_rates, magnitudes)


def _empty_mpc_table():
    return _make_mpc_table([], [], [], [], [], [], [])


def _make_mpc_table(names, numbers, ra, dec, ra_rate, dec_rate, magnitudes):
    table = Table()
    table["Name"] = names
    table["Number"] = numbers
    table["RA"] = np.asarray(ra) * u.deg
    table["DEC"] = np.asarray(dec) * u.deg
    table["RA_rate"] = np.asarray(ra_rate) * (u.arcsec / u.min)
    table["DEC_rate"] = np.asarray(dec_rate) * (u.arcsec / u.min)
    table["sky_speed"] = np.hypot(ra_rate, dec_rate) * (u.arcsec / u.min)
    table["posunc"] = np.full(len(table), np.nan) * u.arcsec
    table["V"] = np.asarray(magnitudes) * u.mag
    table["class"] = np.full(len(table), "", dtype="U1")
    table["sub-class"] = np.full(len(table), "", dtype="U1")
    table["discovery_source"] = np.full(len(table), "MPC MPChecker", dtype="U20")
    return table


def _number_from_name(name):
    parenthesized_designation = re.fullmatch(
        r"\s*\(\s*\d{4}\s+[A-Z]{1,2}\d+[A-Z]?\s*\)\s*", name
    )
    if parenthesized_designation:
        return ""

    parenthesized = re.match(r"^\s*\(\s*(\d+)\s*\)\s+\S", name)
    if parenthesized:
        return parenthesized.group(1)

    numbered = re.match(r"^\s*(\d+)\s+", name)
    if not numbered:
        return ""
    number = numbered.group(1)
    remainder = name[numbered.end():]
    provisional = re.match(r"[A-Z]{1,2}\d*[A-Z]?(?:\s|$)", remainder)
    if len(number) == 4 and 1800 <= int(number) <= 2099 and provisional:
        return ""
    return number


def query_mpc_identification(info: ImageInfo, radius):
    """Find minor planets with MPC MPChecker for the existing catalog pipeline."""
    radius_arcmin = float(u.Quantity(radius).to_value(u.arcmin))
    if not np.isfinite(radius_arcmin) or radius_arcmin <= 0:
        raise ValueError("The MPChecker search radius must be finite and positive")
    if not info.observatory_code:
        raise ValueError("A valid MPC observatory code is required for MPChecker")
    if len(info.observatory_code) != 3:
        raise ValueError("MPC MPChecker requires a three-character observatory code")
    if radius_arcmin > 300:
        raise ValueError("MPC MPChecker supports search radii up to 300 arcminutes")

    radius_arcmin = int(np.ceil(radius_arcmin))
    utc = info.start_obs.utc.datetime
    seconds_today = (
        utc.hour * 3600 + utc.minute * 60 + utc.second + utc.microsecond / 1e6
    )
    centidays = int(round(seconds_today / 864))
    if centidays == 100:
        utc += timedelta(days=1)
        centidays = 0
    day = utc.day + centidays / 100
    params = {
        "year": str(utc.year),
        "month": f"{utc.month:02d}",
        "day": f"{day:.2f}",
        "which": "pos",
        "ra": info.center.ra.to_string(
            unit=u.hourangle, sep=" ", precision=2, pad=True
        ),
        "decl": info.center.dec.to_string(
            unit=u.deg, sep=" ", precision=2, pad=True, alwayssign=True
        ),
        "TextArea": "",
        "radius": str(radius_arcmin),
        # MPChecker requires a magnitude limit; 30.0 is its highest verified
        # accepted value and is not equivalent to an unlimited search.
        "limit": f"{MPC_CHECKER_MAG_LIMIT:.1f}",
        "oc": info.observatory_code,
        "sort": "d",
        "mot": "h",
        "tmot": "s",
        "pdes": "u",
        "needed": "f",
        "ps": "n",
        "type": "p",
    }

    print(
        f"Querying MPC MPChecker around {params['ra']}, {params['decl']} "
        f"on {utc.year}-{utc.month:02d}-{day:.2f} "
        f"(radius {radius_arcmin} arcmin, V <= {MPC_CHECKER_MAG_LIMIT:.1f})...",
        flush=True,
    )
    with requests.Session() as session:
        session.headers["User-Agent"] = MPC_CHECKER_USER_AGENT
        form = session.get(MPC_CHECKER_FORM_URL, timeout=(15, 30))
        form.raise_for_status()
        response = session.post(
            MPC_CHECKER_URL,
            data=params,
            headers={
                "Referer": form.url,
                "Origin": "https://minorplanetcenter.net",
            },
            timeout=(15, 90),
        )
        response.raise_for_status()

    table = _mpchecker_table(response.text)
    print(f"MPC MPChecker found {len(table)} minor planets.", flush=True)
    return table
