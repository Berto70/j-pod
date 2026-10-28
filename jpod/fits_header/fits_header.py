
"""Read the observatory code and the observation time from a FITS header.
 
Header keywords differ from one survey to another, so for each quantity there
is a short list of candidates (below), tried in order. Anything that cannot be found produces a
clear message or a warning. To support a new dataset, add its keyword to the
right list. 

The time found in the header can refer to the start, the middle or the end of the exposure. 
From it, and from the exposure time, the start and the end of the exposure are computed (get_obs_times).
"""
 
import re
import astropy.units as u
from astropy.time import Time
from .obscodes import find_code
 
# ---------------------------------------------------------------- candidates
# Time of the exposure: (keyword, how to read it, meaning), in order of preference.
# kind:  iso= date/time string, jd= Julian date, mjd= modified Julian date.
# meaning: what the time refers to when the keyword itself says so (start, mid, end), or None when it must be found from the header comment

TIME_KEYWORDS = (
    ("DATE-AVG", "iso", "mid"),
    ("MJD-AVG", "mjd", "mid"),
    ("DATE-BEG", "iso", "start"),
    ("MJD-BEG", "mjd", "start"),
    ("DATE-OBS", "iso", None),
    ("OBSJD", "jd", None),
    ("OBSMJD", "mjd", None),
    ("MJD-OBS", "mjd", None),
    ("SHUTOPEN", "iso", "start"),
)
 
# Exposure time in seconds.
EXPTIME_KEYWORDS = ("EXPTIME", "EXPOSURE", "EXPTIM")
 
# Keywords that directly hold the MPC observatory code.
OBS_CODE_KEYWORDS = ("OBSCODE", "MPCCODE", "OBS-CODE")
 
# Used (with a warning) when no code can be found: the centre of the Earth.
GEOCENTRIC_CODE = "500"
 
# MPC codes are 3 digits (e.g. 500) or one letter + 2 digits (e.g. C89, J75).
OBS_CODE_PATTERN = re.compile(r"^(\d{3}|[A-Za-z]\d{2})$")
 
# What the start time refers to, and the fraction of the exposure to add to
# reach the end of the exposure.
TIME_REFERENCES = {"start": 1.0, "mid": 0.5, "end": 0.0}

# Words in a header comment that say what the time refers to
COMMENT_PATTERNS = {
    "start": r"\b(start\w*|begin\w*|open\w*)\b",
    "mid": r"\b(mid\w*|middle|average|avg|cent(?:er|re))\b",
    "end": r"\b(end\w*|stop\w*|close\w*)\b",
} 
 
# ---------------------------------------------------------------- code
def get_observatory_code(header, override=None, obscodes=None):
    """Return (code, source); source says where the code came from.
 
    Order: 
    1) 'override'; 
    2) a header keyword in OBS_CODE_KEYWORDS;
    3) the MPC list 'obscodes', by coordinate or name;
    4) geocentric code, with a warning.
    """
    if override:
        return str(override).strip().upper(), "set manually"
 
    for key in OBS_CODE_KEYWORDS:
        if key in header:
            value = str(header[key]).strip()
            if OBS_CODE_PATTERN.match(value):
                return value.upper(), f"header keyword {key}"
    reason = "no list of observatory codes was given"
    if obscodes:
        code, message = find_code(obscodes, header)
        if code:
            return code, message
        reason = message
    
    print(f"WARNING: no observatory code found ({reason}). "
          f"Using the geocentric code {GEOCENTRIC_CODE}: positions of nearby "
          "objects will be inaccurate. Set OBSERVATORY_CODE in main.py.")
    return GEOCENTRIC_CODE, "NOT FOUND, geocentric default"
 
 
# ---------------------------------------------------------------- time
def _read_time(value, kind):
    if kind == "jd":
        return Time(float(value), format="jd", scale="utc")
    if kind == "mjd":
        return Time(float(value), format="mjd", scale="utc")
    return Time(str(value).strip(), scale="utc")


def _comment(header, key):
    try:
        return str(header.comments[key]).strip()
    except (AttributeError, KeyError, IndexError):
        return ""

def _meaning_from_comment(comment):
    """
    'start', 'mid' or 'end' if the comment says it clearly, else None.
    """
    text = comment.lower()
    found = [name for name, pattern in COMMENT_PATTERNS.items()
             if re.search(pattern, text)]
    return found[0] if len(found) == 1 else None

def get_header_time(header, override = None):
    """
    Return (time, keyword, meaning, comment) for the time in the header.
    meaning is what the keyword itself says (or None);
    comment is the header comment of the keyword;
    'override' is a time string (UTC) that replaces the header;
    Raises ValueError if nothing usable is found, because without a time the asteroids' positions cannot be predicted.
    """
    if override:
        return Time(str(override).strip(), scale ="utc"), "set manually", None, ""
    for key, kind, meaning in TIME_KEYWORDS:
        if key not in header:
            continue
        try:
            value = header[key]
            # Old-style header: date in DATE-OBS, time in TIME-OBS
            if key == "DATE-OBS" and "T" not in str(value) and "TIME-OBS" in header:
                value = f"{str(value).strip()}T{str(header['TIME-OBS']).strip()}"
            return _read_time(value, kind), key, meaning, _comment(header, key)
        except (ValueError, TypeError):
            continue 
    tried = [k for k,_,_ in TIME_KEYWORDS]
    raise ValueError(
        f"No usable observation time in the header (tried {tried})."
        "Set DATE_OBS in main.py, e.g. DATE_OBS = '2018-12-30T01:58:54' "
        "(UTC), or add the right keyword to TIME_KEYWORDS in fits_header.py."
    )
 

def get_exposure_time(header):
    """Return (exposure time in seconds as Quantity, keyword) or (None, None)."""
    for key in EXPTIME_KEYWORDS:
        if key in header:
            try:
                value = float(header[key])
            except (TypeError, ValueError):
                continue
            if value > 0:
                return value * u.s, key
    return None, None
 
 
def get_obs_times(header, time_reference=None, date_obs_override=None):
    """Return (date_obs, start_obs, end_obs, exptime, source).
 
    date_obs : the time found in the header.
    start_obs: time at which the exposure starts (UTC)
    end_obs  : time at which the exposure ends (UTC) 
    exptime  : exposure time (Quantity) or None.
    source   : text describing what was used, for the log.
    time_reference : None (automatic) or 'start', 'mid', 'end' to force what the header time refers to.
    In automatic mode it is decided by the keyword itself (e.g. DATE-AVG is the middle), then by the header comment,
    and finally assumed to be the start (FITS convention).
    If the exposure time is not found a warning is printed and start_obs = end_obs = date_obs.
    """
    if time_reference not in (None, *TIME_REFERENCES):
        raise ValueError(f"time_reference must be None or one of {list(TIME_REFERENCES)}")
 
    timesys = str(header.get("TIMESYS", "UTC")).strip().upper()
    if timesys != "UTC":
        print(f"WARNING: TIMESYS is {timesys!r}, but the time is treated as UTC.")
 
    date_obs, key, keyword_meaning, comment = get_header_time(header, date_obs_override)
    if time_reference:
        ref, why = time_reference, "set manually"
    elif keyword_meaning:
        ref, why = keyword_meaning, f"by definition of {key}"
    else:
        ref = _meaning_from_comment(comment)
        if ref:
            why = f"from the header comment {comment!r}"
        else:
            ref = "start"
            note = f"header comment: {comment!r}" if comment else "no header comment"
            why = f"ASSUMED, as in the FITS convention; {note}"
    origin = key if key == "set manually" else f"keyword {key}"
    source = f"{origin}, taken as the {ref} of the exposure ({why})"
    exptime, exp_key = get_exposure_time(header)
    if exptime is None:
        print(f"WARNING: no exposure time found (tried {list(EXPTIME_KEYWORDS)}). "
              "Start and end of the exposure are both set to the header time.")
        return date_obs, date_obs, date_obs, None, source + "; no exposure time"
    end_obs = date_obs + TIME_REFERENCES[ref]*exptime
    start_obs = end_obs - exptime
    return date_obs,start_obs, end_obs, exptime, source + f"; exposure from {exp_key}"


    
   







