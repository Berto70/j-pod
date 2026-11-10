# J-POD pipeline guide

This guide follows the order in which one FITS image is processed by
`jpod.main.pipeline`. When a directory is selected, `main` calls this pipeline
once for each FITS file.

## 1. Load the FITS image and resolve metadata

This section traces what happens from the time [`pipeline()`](../jpod/main.py#L137)
starts processing one FITS file until the magnitude threshold is ready for
asteroid detection. It deliberately follows the code order. Appendix A gives a
more granular walkthrough of the image-analysis and catalog-matching helper
algorithms.

### 1.1 Entry point and call order

The command-line entry point in `main.py` selects one file or a directory of
files and calls `pipeline()` once for each image. For each file, the relevant
photometry path is:

```text
main()
  -> pipeline(fits_file, args, output_dir)
       -> load_image(...)
       -> image_orientation(...)
       -> measure_image(info)
       -> determine_zero_point(info, measurements, catalog_dir)
       -> calculate_limiting_magnitude(...)  [V filter and usable ZP only]
       -> use v_lim in add_detection_probability(...) and plots
```

The order matters. `measure_image()` is called **before**
`determine_zero_point()`, even if the FITS header already contains a zero
point. It produces the background and FWHM needed by the later limit
calculation and also detects/measures stars for a possible catalog
calibration. A usable header zero point makes `determine_zero_point()` return
without opening the catalog, but it does not skip the already-completed image
measurement pass.

### 1.2 `load_image()`: pixels, WCS, and metadata

[`load_image()`](../jpod/image_geometry.py#L40) in `image_geometry.py` opens
the FITS file and reads the **primary HDU's** header and pixel array. The
returned `ImageInfo` object is the shared bundle of information used by later
steps. At a high level it contains:

- the image data array and its width and height;
- a WCS (World Coordinate System) made from the FITS header;
- sky coordinates for the image center and four corners;
- pixel scale in arcseconds per pixel;
- the observation time, exposure start/end, and exposure duration;
- the resolved observatory code and where it came from;
- selected header values: `FWHM`, `ZP`, `MAGZERO`, `PHOTZP`, `SATURATE` or
  `SATLEVEL`, `EGAIN` or `GAIN`, and `FILTER`; and
- `BUNIT` plus relevant unit-related card comments, used to classify pixels
  as integrated ADU/DN, integrated electrons, ADU/DN per second, electrons per
  second, or unknown.

The WCS is essential for catalog calibration. Catalog stars are stored at sky
coordinates (right ascension and declination); the WCS projects those
coordinates to image pixels so the program can match catalog and detected
stars.

#### Observatory-code helper calls

`load_image()` asks `load_obscodes_safe()` to read the MPC observatory-code
list, then calls `get_observatory_code()`:

1. `get_observatory_code()` first checks header keywords `OBSCODE`,
   `MPCCODE`, and `OBS-CODE`.
2. If those do not provide a valid code, it calls `find_code()` in
   `fits_header/obscodes.py`. The list can be matched against the site's
   longitude/latitude or, if needed, telescope/site names.
3. If there is no unambiguous list match, the user-supplied `--observatory`
   value is used when provided.
4. Otherwise, the code uses geocentric MPC code `500` and prints a warning.

This code identifies the observing site for ephemeris calculations; it is
separate from image photometry.

#### Observation-time helper calls

`load_image()` also calls `get_obs_times()`:

1. `get_header_time()` searches an ordered list of common timestamp keywords
   such as `DATE-AVG`, `DATE-BEG`, `DATE-OBS`, `OBSJD`, and `MJD-OBS`.
2. The time's meaning (exposure start, middle, or end) is inferred from the
   keyword definition or its FITS comment. `DATE-OBS` with no clear comment
   defaults to start, following the documented FITS convention. Module-level
   overrides in `main.py` can replace this behavior.
3. `get_exposure_time()` searches `EXPTIME`, `EXPOSURE`, and `EXPTIM`.
4. From the selected reference time and duration, the helper calculates the
   exposure start and end. If duration is missing, it warns and sets both
   start and end equal to the selected time.

The pipeline prints `describe_time(info)` so the chosen timestamp, its
interpretation, and exposure interval can be checked in the terminal.

#### Other FITS cards collected

The code reads finite numeric zero-point values from `ZP`, `MAGZERO`, and
`PHOTZP`, in that precedence order. This step only parses their numbers; it
does not prove that a value uses the right filter or flux convention. The
first valid saturation card among
`SATURATE` and `SATLEVEL` can be used to exclude saturated stars. For gain, `EGAIN` is tried before `GAIN`. The loader classifies explicit
`BUNIT` values and relevant comments. Generic “counts” does not tell ADU/DN
from electrons, and conflicting clues are treated as unknown rather than
guessed. If explicit units are absent, comments on `PEDESTAL`, `CBLACK`, or
`CWHITE` that explicitly identify ADU/DN can support an inference of
integrated ADU/DN; a gain comment such as “e-/ADU” corroborates this but is
not sufficient by itself. Explicit image-unit metadata takes precedence.
A plain ADU/DN or electron label is treated as integrated data; an explicit
`/s`, `/sec`, or inverse-second suffix identifies a rate image.
ADU/DN sky-noise calculations require a positive gain; electron data do not
apply the gain a second time. Header zero points are still interpreted using
the existing per-second-flux convention, in the image's native data unit;
this convention should be verified for individual camera/software headers.

### 1.3 `measure_image()`: one shared image-measurement pass

Back in `pipeline()`, J-POD calls
[`measure_image(info)`](../jpod/image_analysis.py#L45). This function in
`image_analysis.py` is used both to report stellar FWHM and to provide
measurements to zero-point and limiting-magnitude calculations. Its return
value, `ImageMeasurements`, contains:

- `fwhm_pixels`: the selected representative stellar width;
- `background`: the spatial background map in the image's native pixel units;
- `noise`: a robust residual-scatter estimate used for detecting candidate
  stars; and
- `stars`: vetted `ImageStar` records with pixel positions, aperture fluxes,
  profile widths, and WCS-derived sky coordinates.

In order, `measure_image()`:

1. Converts the data to floating-point and rejects non-2-D images or images
   with no finite pixels.
2. Fills non-finite pixels only for the background-estimator input, while
   retaining a mask so they are excluded from measurements.
3. Calls Photutils `Background2D` with 50-by-50-pixel boxes, a 3-by-3 median
   filter, and `MedianBackground`. This returns a map of the local background.
4. Subtracts that map to form a residual image. It estimates a robust
   residual scatter with \(1.4826 \times \mathrm{median}(|x-\mathrm{median}(x)|)\).
   This scatter helps identify stars; it is **not** the aperture sky noise
   used in the limiting-magnitude equation.
5. Finds local maxima using a 9-by-9 maximum filter, retaining peaks above
   five times the residual scatter. It processes no more than the 100
   brightest candidates.
6. For each candidate, examines an 11-by-11-pixel cutout. Positive
   background-subtracted values estimate the centroid and profile moments.
   The moments give a profile width and elongation; overly elongated, low-SNR,
   saturated, or flat-topped candidates are rejected.
7. Calls
   [`_reject_blended_profiles()`](../jpod/image_analysis.py#L266) to remove
   detected profiles with another candidate within twice their local FWHM.
8. Calls
   [`_robust_median_fwhm()`](../jpod/image_analysis.py#L281) to find a
   representative profile width. It iteratively rejects widths more than
   three scaled MADs from the median, for up to three rounds.
9. Calls [`_select_fwhm()`](../jpod/image_analysis.py#L246) to decide whether
   to use that measured value or a header FWHM. A finite, positive header
   width is accepted unless it is more than a factor of two larger or smaller
   than the measured width. On such a mismatch it prints a note and uses the
   image-measured width.
10. Measures each retained star's **aperture flux**: it sums valid,
    background-subtracted pixels within a circular aperture whose radius is
    the selected FWHM. Non-positive fluxes and edge-truncated apertures are
    skipped. The brightest 50 resulting stars are retained for catalog
    matching.

If there are no candidate peaks, the residual scatter is unusable, or all
candidate profiles fail the quality checks,
[`_measurements_without_stars()`](../jpod/image_analysis.py#L232) preserves a
valid header FWHM when one exists and returns an empty star list. Without a
valid header FWHM, it returns the background/noise measurements with
`fwhm_pixels=None`; this lets the rest of the pipeline continue. The catalog
calibration and automatic limiting-magnitude calculation report that they
cannot proceed without stars or FWHM. The user-provided `--vlim` remains in
effect, trailing-loss columns are unavailable, and magnitude probabilities
use the unadjusted asteroid `V` magnitude.

`_reject_blended_profiles()` also handles an empty input list directly, so it
does not pass a malformed array to `cdist()`.

### 1.4 `determine_zero_point()`: header first, catalog second

`pipeline()` passes `ImageInfo`, `ImageMeasurements`, and the resolved
catalog path to
[`determine_zero_point()`](../jpod/photometric_calibration.py#L44). It uses
this precedence:

1. If a valid numeric `ZP` exists, use it. If either `MAGZERO` or `PHOTZP`
   also exists and differs, issue a Python warning; `ZP` still wins.
2. Otherwise, if a valid numeric `MAGZERO` exists, use it. If `PHOTZP` also
   exists and differs, issue a Python warning; `MAGZERO` still wins.
3. Otherwise, if a valid numeric `PHOTZP` exists, use it.
4. Only when no header zero point exists does the function consider a
   catalog-derived zero point. Catalog calibration is restricted to filters
   recognized by `is_v_filter()` (for example `V`, `JOHNSON V`, or
   `BESSELL V`) and recognized image units.
5. It checks exposure time, celestial WCS, shared measurements, catalog
   directory, and the number of detected stars. A catalog-derived calibration
   requires at least five valid matches.

For catalog matching, the helper sequence is:

```text
determine_zero_point()
  -> _catalog_files_for_stars()
       -> _cell_for_coordinate()
  -> _read_v50_files()
  -> WCS world_to_pixel()
  -> _match_stars()
  -> robust median / 3-MAD outlier rejection
```

`_catalog_files_for_stars()` selects only ASTAP sky cells around the detected
stars, adding surrounding points so a match near a cell boundary is not
missed. `_cell_for_coordinate()` converts each sky position to ASTAP's
one-based ring/cell numbering. `_read_v50_files()` decodes the selected
binary V50 records into arrays of RA, declination, and Johnson V magnitude.
The WCS projects those catalog positions to pixels; `_match_stars()` makes
nearest-neighbor, one-to-one matches within the pixel tolerance.

For each retained match, integrated-image flux is divided by exposure time;
rate-image flux is used directly. The individual per-second zero point is

\[
ZP_i = V_i + 2.5\log_{10}(F_{i,\rm rate}),
\]

where \(V_i\) is the catalog magnitude and \(F_i\) is the measured,
background-subtracted aperture flux in the image's native units;
\(F_{i,\rm rate}\) is that flux per second in the same native units. Outliers
are clipped using the median and scaled median absolute deviation (MAD). The
final image zero point is the median of the remaining individual estimates.
The reported robust scatter describes their spread; it is not yet a full
uncertainty estimate.

The catalog is **not opened** when a usable header zero point is found. The
image-measurement pass still ran first, as described above.

### 1.5 `calculate_limiting_magnitude()`: background to one V limit

If a zero point is available and the filter is recognized as V,
`pipeline()` calls
[`calculate_limiting_magnitude()`](../jpod/photometric_calibration.py#L184).
It takes the median \(B\) of the spatial background map and converts it to a
background rate \(b\) in the image's native unit per second per pixel. For
integrated images \(b=B/t_{\rm exp}\); for rate images \(b=B\). Let \(q\) be
electrons per native image unit: the header gain for ADU/DN, and 1 for
electrons. For an aperture of area \(A_{\rm pix}\), the sky-only variance in
the aperture flux **rate** is:

\[
A_{\rm pix} \approx \pi\,\mathrm{FWHM}_{\rm pixels}^{\,2},
\qquad
\sigma_{\rm sky,rate}^{2}
= \frac{A_{\rm pix}\,b}{q\,t_{\rm exp}},
\]
\[
V_{\rm lim}
= ZP - 2.5\log_{10}\!\left(
\mathrm{SNR}\,\sigma_{\rm sky,rate}\right).
\]

This is a sky-Poisson-only model. For ADU/DN, automatic \(V_{\rm lim}\) is
unavailable if no positive `EGAIN` or `GAIN` is present; the program does not
assume unit gain. If the image units are unknown, automatic catalog
calibration and \(V_{\rm lim}\) are unavailable and the user-provided
`--vlim` remains in effect. The SNR threshold defaults to 3 and can be changed
with `--snr`.

The calculated value replaces `args.vlim` for the magnitude-probability
calculation and plots only if all required conditions succeed. A non-V
filter, unavailable zero point, invalid background, or other calculation
failure keeps the user-provided `--vlim` fallback. The filter check is made
in `pipeline()` even for a header zero point; currently the automatic V limit
is used only when the image is explicitly recognized as V band.

### 1.6 Which functions do what?

| Function | File | Role in this part of the pipeline |
|---|---|---|
| `pipeline()` | [`main.py`](../jpod/main.py#L137) | Orchestrates image loading, measurements, calibration, limit calculation, and later detection. |
| `load_image()` | [`image_geometry.py`](../jpod/image_geometry.py#L40) | Reads primary image/header, classifies pixel units, and constructs the `ImageInfo` context. |
| `get_observatory_code()` | [`fits_header/fits_header.py`](../jpod/fits_header/fits_header.py#L59) | Resolves the MPC observing-site code. |
| `get_obs_times()` | [`fits_header/fits_header.py`](../jpod/fits_header/fits_header.py#L159) | Interprets header time and calculates exposure start/end. |
| `measure_image()` | [`image_analysis.py`](../jpod/image_analysis.py#L45) | Estimates background/noise, FWHM, candidate star profiles, and aperture fluxes. |
| `determine_zero_point()` | [`photometric_calibration.py`](../jpod/photometric_calibration.py#L44) | Returns a header zero point or calculates one from V50 matches. |
| `calculate_limiting_magnitude()` | [`photometric_calibration.py`](../jpod/photometric_calibration.py#L184) | Turns unit-aware background, FWHM, gain, exposure, zero point, and SNR into \(V_{\rm lim}\). |
| `add_detection_probability()` | [`detection.py`](../jpod/detection.py#L57) | Uses the selected limit to compute magnitude and combined detection probabilities. |

For the detailed operations inside `measure_image()` and the V50 reader/matcher,
see [Appendix A](#appendix-a-photometry-code-walkthrough).

## 2. Choose a small-body discovery provider

The `--provider` CLI option selects one provider for this run:

- `--provider mpc` queries MPC MPChecker.
- `--provider skybot` queries SkyBoT.

The default is `skybot`. The pipeline computes a circular search radius that
contains the image, plus the configured margin, then passes that quantity to
the selected adapter. Each adapter converts the radius to the provider's
expected units.

Both provider requests use the same field centre, search radius, and resolved
observatory code, but they are not otherwise identical:

### MPC MPChecker (`--provider mpc`)

- MPChecker searches minor planets. The adapter currently requests its
  minor-planet result type and sends the resolved code in the form's `oc` field.
- MPChecker requires a magnitude limit. J-POD currently sends `V <= 30`; this
  is a provider-side query limit, not the image's `--vlim` detection threshold.
- MPChecker accepts the requested epoch to centiday precision. J-POD rounds
  the image start time to the nearest 0.01 day for its query and records the
  resulting rounded time as `provider_epoch`. Therefore its returned positions
  and rates need not refer to the exact image start.
- MPChecker reports motion components in arcseconds per hour with coarser
  precision than SkyBoT; the adapter parses those values into the shared
  tangent-plane rate convention. It does not provide positional uncertainty
  or object dynamical class in the parsed result.

### SkyBoT (`--provider skybot`)

- J-POD asks SkyBoT for asteroids and comets, but not planets.
- SkyBoT receives the exact UTC image-start epoch.
- The adapter passes `position_error=0 arcsec`, disabling SkyBoT's default
  120-arcsecond positional-error cutoff. This retains poorly constrained
  objects too; their reported positions can be far from Horizons predictions,
  so inspect `posunc` and the Horizons uncertainty ellipse when comparing them.
- SkyBoT provides positional uncertainty and object type information when
  available. Its uncertainty can be extremely large for poorly constrained
  objects; disabling the cutoff is not a claim that those predictions are
  precise.

### Example: J75 image

For `data/J75_01_20110720204510.fits`, the header gives observatory code `J75`
and an exposure from `2011-07-20 20:45:10` to `20:46:40 UTC`. Both provider
queries use the same approximately `1.067 deg` search radius. SkyBoT searches
at `20:45:10 UTC`, while MPChecker's centiday-rounded search epoch is
`20:38:24 UTC`, 6 minutes 46 seconds earlier. The two providers' values should
therefore not be compared as if they were measured at exactly the same instant.
Horizons is queried at each provider's own epoch so the provider-to-Horizons
comparison is time-aligned.

## 3. Normalize provider results

Both adapters return an Astropy table following `providers/schema.py`:

- RA and DEC are quantities in degrees.
- `RA_cosdec_rate` and `DEC_rate` are tangent-plane east/north rates in
  arcseconds per hour. The RA component already includes cos(DEC).
- `sky_speed` is the hypotenuse of those tangent-plane rate components, also
  in arcseconds per hour. No extra cosine is applied.
- `V` is a magnitude quantity.
- `posunc` is in arcseconds where the provider supplies it; MPChecker's
  unavailable value is NaN.
- The `class` and `sub-class` fields are filled where provided; MPChecker's
  unavailable values are empty.
- `provider` and `provider_epoch` identify the source and the epoch associated
  with its reported position/rates.

The selected provider's `Name` and `Number` are retained for resolving each
candidate in Horizons.

## 4. Enrich candidates with JPL Horizons

`enrich_with_horizons` in `horizons.py` queries Horizons using candidate names
or numbers. It asks for the provider's discovery epoch plus the exposure start
and end epochs. Querying the discovery epoch allows the provider's coordinates
and rates to be compared with Horizons at the same time, even when MPChecker's
rounded epoch differs from the image start. The start and end epochs supply
the track used by the image pipeline.

Horizons rates are converted into the shared convention of
`RA_cosdec_rate`, `DEC_rate`, and `sky_speed` in arcseconds per hour. The
Horizons comparison values remain in separate in-memory columns. The final
working columns `RA`, `DEC`, `RA_end`, `DEC_end`, and the rate columns use the
Horizons track when Horizons supplies a complete track and both rate
components. Otherwise, they retain the provider position/rates, and the end
position is projected from the provider rates.

The `source` column says which source supplied the final working position and
rates: either the provider's name (for example, `SkyBoT` or `MPC MPChecker`) or
`JPL Horizons`. The original provider position, rate, apparent magnitude, and
uncertainty are retained separately in memory for comparisons. Horizons
position/rate comparison values are at the provider's search epoch; Horizons
absolute magnitude `H` and uncertainty ellipse values are at exposure start.
`H` is absolute magnitude, unlike the providers' apparent `V`.

The providers' results can differ for reasons other than epoch: MPChecker
imposes its `V <= 30` query limit and reports minor planets, while SkyBoT is
asked for asteroids and comets and returns positional uncertainty and type
metadata. SkyBoT's disabled uncertainty cutoff may expose very poor orbits.
Provider names/number formatting and catalog coverage can also differ, so a
comparison should match normalized designations where possible and keep
provider-only objects visible.

## 5. Project onto the image and filter

`add_ellipse_image_angle` converts the Horizons ellipse orientation to the
image's pixel-coordinate orientation. `filter_in_image` projects the object's
start and end positions through the FITS WCS and keeps candidates whose
position or uncertainty ellipse overlaps the image. When Horizons did not
provide an ellipse, the positional uncertainty available from the provider is
used as a circular uncertainty.

The pipeline then computes trail length from `sky_speed` and exposure time.
Because speed is in arcseconds per hour, Astropy handles conversion against
the exposure duration to obtain the angular and pixel trail lengths.

## 6. Compute detection probabilities

`add_detection_probability` combines:

- `P_pos`: probability that the object's uncertain position is within the
  image over the exposure.
- `P_mag`: probability that its apparent V magnitude is brighter than the
  configured limiting magnitude.
- `P_det`: the product of those two probabilities.

The CLI's `--pmin` threshold marks each row as detectable or not. Results are
sorted by `P_det` before plotting and saving.

## 7. Plot and save results

The pipeline creates candidate and survivor plots, and optional zoom plots.
It then writes the compact CSV selection from `OUTPUT_COLUMNS` in
`jpod/config.py`. Provider identification fields, epoch details, and
provider-versus-Horizons comparison columns remain available in the in-memory
table but are intentionally omitted from the CSV to keep the saved table
readable.

## Appendix A. Photometry code walkthrough

This appendix expands Section 1 into a call-by-call account. File links point
to the relevant functions in the source tree. Read the functions in the
sequence below while following one image through `pipeline()`.

### A.1 From the command line to one FITS file

[`main()`](../jpod/main.py#L333) parses the command-line arguments and chooses
the working directory. `resolve_path()` turns the selected data and output
paths into paths relative to that working directory (unless they are already
absolute). The file selection is either the explicitly named `--file` or all
`.fits` files in the selected input directory. For each file, `main()` calls
[`pipeline()`](../jpod/main.py#L137).

The photometry-specific portion of `pipeline()` has three distinct products:

1. `measure_image()` produces image measurements, including a background map,
   FWHM, and measured stars.
2. `determine_zero_point()` produces either a header/catalog zero point or a
   reason it could not do so.
3. If the image filter is recognized as V and the zero point is available,
   `calculate_limiting_magnitude()` produces an automatic V limit. Otherwise
   the configured `--vlim` remains the value used downstream.

The image-analysis pass is shared, but the result objects remain conceptually
different: measured pixel/stars data are not themselves a calibration, and a
zero point is not itself a limiting magnitude.

### A.2 Exact work performed by `load_image()`

In [`load_image()`](../jpod/image_geometry.py#L40):

1. `fits.open()` opens the FITS file. The code copies `hdul[0].data` from the
   **primary HDU** and retains `hdul[0].header`. It does not search alternate
   HDUs for image arrays.
2. `NAXIS1` and `NAXIS2` supply width and height. `WCS(header)` constructs the
   pixel-to-sky coordinate transform from the FITS WCS cards.
3. `SkyCoord.from_pixel()` converts the geometric center and the four image
   corners into sky coordinates. The corners are used later to size a sky
   search around the field.
4. `proj_plane_pixel_scales(w.celestial)` estimates the celestial pixel scale;
   the first axis is stored in arcseconds per pixel. The code currently
   treats that as a representative scale for both axes.
5. `load_obscodes_safe()` reads the MPC observatory list, then
   `get_observatory_code()` resolves the site code.
6. `get_obs_times()` determines the timestamp, its reference within the
   exposure, and the exposure interval.
7. The loader extracts the supported photometry-related header cards,
   classifies units from `BUNIT` and relevant comments, then returns all of
   this as one `ImageInfo`. `_classify_image_units()` recognizes explicit
   ADU/DN, electrons, and per-second forms. It considers the BUNIT value and
   its FITS comment, selected unit-description cards, and comments that
   explicitly describe image, pixel, or data units. A gain-card comment such
   as “gain in e-/ADU” describes the conversion factor, not necessarily the
   stored image values. If explicit unit metadata is absent, ADU/DN comments
   on `PEDESTAL`, `CBLACK`, or `CWHITE` can identify integrated ADU/DN, with
   gain comments used only as corroborating evidence. Explicit image-unit
   metadata takes precedence over these secondary clues.

The data pixels are not calibrated or rescaled by `load_image()`. Loading
metadata such as a gain does not apply the gain to the pixel array. Unit
classification describes the existing values; it does not transform them.

### A.3 Observatory lookup, one helper level deeper

[`get_observatory_code()`](../jpod/fits_header/fits_header.py#L59) follows a
fixed priority:

1. Try `OBSCODE`, `MPCCODE`, and `OBS-CODE`; accept a code matching three
   digits or one letter followed by two digits.
2. If no direct code is usable, call
   [`find_code()`](../jpod/fits_header/obscodes.py#L148).
3. `find_code()` attempts a location match using longitude and latitude
   keywords; [`site_xyz()`](../jpod/fits_header/obscodes.py#L48) converts a
   geodetic location into the MPC list's Earth-centered representation. If a
   location match is unavailable, the code can compare significant words in
   `TELESCOP`/`OBSERVAT` to site names. Ambiguous name matches at different
   locations are not guessed.
4. If there is no unique list match, use the `--observatory` argument, if
   supplied; otherwise print a warning and use geocentric code `500`.

`load_obscodes_safe()` catches missing/unreadable-list errors and reports
them, allowing the direct header code or manual fallback still to be used.
The observatory lookup supports ephemeris calculations; it does not affect
the photometric zero point.

### A.4 Time interpretation, one helper level deeper

[`get_obs_times()`](../jpod/fits_header/fits_header.py#L159) asks
[`get_header_time()`](../jpod/fits_header/fits_header.py#L117) to scan
`TIME_KEYWORDS` in order. `get_header_time()` delegates parsing to
[`_read_time()`](../jpod/fits_header/fits_header.py#L94) for ISO strings,
Julian dates, or modified Julian dates. A recognized keyword can define a
time as the exposure start, middle, or end. If the keyword does not define
it, [`_meaning_from_comment()`](../jpod/fits_header/fits_header.py#L108)
looks for words such as “start”, “mid”, or “end” in the FITS comment. A
configured `TIME_REFERENCE` overrides this; otherwise an unclear time is
assumed to be the exposure start.

[`get_exposure_time()`](../jpod/fits_header/fits_header.py#L146) searches
`EXPTIME`, `EXPOSURE`, and `EXPTIM`, accepting a positive numeric value and
expressing it in seconds. From that duration and the timestamp reference,
`get_obs_times()` calculates `start_obs` and `end_obs`. Thus a header
timestamp at mid-exposure is not mistakenly treated as the start when the
keyword or comment identifies its meaning.

### A.5 How `measure_image()` creates the background map and stars

The entry point is
[`measure_image()`](../jpod/image_analysis.py#L45). Its significant
intermediate data are:

```text
data -> background map -> residual image -> candidate profiles
     -> selected FWHM -> aperture fluxes and star sky coordinates
```

#### Background and detection-noise estimate

The data array is converted to float. A `valid` mask marks finite pixels.
Non-finite pixels are replaced by the median only in the temporary array
passed to Photutils; the mask prevents them from being treated as real pixels.
`Background2D` estimates a local background in boxes no larger than 50 pixels
on each axis, applies a 3-by-3 filter, and uses `MedianBackground`.

The residual is `safe_data - background`. The program computes a scaled MAD
from finite residual pixels:

\[
\mathrm{noise}_{\rm detect}
=1.4826\,\mathrm{median}(|r-\mathrm{median}(r)|).
\]

This is a robust estimate used for detecting and qualifying stars. Later, the
sky-only limit uses the background-map level and a Poisson model instead;
these two quantities must not be conflated.

#### Candidate detection and profile measurements

`maximum_filter()` with a 9-by-9 footprint identifies local residual maxima.
Only peaks greater than five times `noise_detect` remain, and at most the 100
brightest candidates are examined. Each candidate must be far enough from the
image edge to fit an 11-by-11 cutout.

For each cutout:

- Saturation is checked against `SATURATE`/`SATLEVEL` if one was found.
- Positive residual values are used as moment weights to estimate a
  subpixel centroid. Negative residual values are excluded from centroid
  weights but remain part of the later flux sum.
- The code measures how those weighted pixels spread around the centroid in
  the horizontal and vertical directions. It stores that information in a
  symmetric 2-by-2 covariance matrix:
  \(\begin{bmatrix}\mathrm{spread}_x^2 & \mathrm{tilt}\\
  \mathrm{tilt} & \mathrm{spread}_y^2\end{bmatrix}\).
  The diagonal entries describe spread along the image's x and y axes; the
  off-diagonal entries describe whether the profile is tilted (whether
  pixels displaced in x also tend to be displaced in y). The matrix's
  eigenvectors point along the profile's two principal directions, and its
  eigenvalues are the variances along those directions. Their square roots
  are therefore Gaussian-equivalent widths, in pixels, along the minor and
  major axes. They are standard-deviation-like widths, not full edge-to-edge
  diameters. The code combines them with a geometric mean and converts that
  result to FWHM using the Gaussian conversion factor below. This is a
  moment-based approximation, not a fit of a full two-dimensional star
  profile.
- Profiles with an axis ratio above 1.3 are rejected as too elongated.
- A geometric-mean width is converted to FWHM using
  \(2\sqrt{2\ln 2}\approx2.35482\), the Gaussian standard-deviation-to-FWHM
  factor.
- An approximate profile SNR must be at least 20.
- If no saturation threshold was supplied, a flat-topped 3-by-3 core is
  rejected as a possible clipped source.

These are pragmatic star-selection rules, not a full PSF fit. Each threshold
can affect which stars contribute to FWHM and calibration.

#### Blend rejection and FWHM selection

`_reject_blended_profiles()` computes pairwise distances and rejects a
profile if another detected profile is within twice the larger of their
individual profile widths. This avoids obviously overlapping stars whose
widths and aperture sums could be contaminated.

`_robust_median_fwhm()` computes a median profile width and applies up to
three rounds of 3-scaled-MAD rejection. Its median is then compared with the
header FWHM by `_select_fwhm()`:

- no header value: use the measured median;
- valid positive header value within a factor of two: use the header value;
- header value more than 2x or less than 0.5x the measured value: print a
  note and use the measured median.

There is a special fallback. If no usable profiles are found, or the residual
scatter cannot support detection, `_measurements_without_stars()` uses a
valid positive header FWHM when present. Without one it returns
`fwhm_pixels=None`, while preserving the background and detection-noise
measurements, so later pipeline stages can use their documented fallbacks
instead of stopping.

#### Aperture flux and star records

For every remaining profile, the code puts a circular aperture at its
subpixel centroid. Its radius is the selected FWHM. It sums residual values
for valid pixels whose centers fall inside that circle:

\[
F_{\rm aperture} =
\sum_{(x,y)\ {\rm inside\ aperture}}
(I_{x,y}-B_{x,y}).
\]

The sum must be finite and positive, and the aperture must fit inside the
image. `SkyCoord.from_pixel()` maps the centroid through the WCS to RA and
Dec. The resulting `ImageStar` carries x/y, aperture flux, peak residual,
individual profile FWHM, RA, and Dec. Stars are sorted by flux, and at most
the brightest 50 are returned in `ImageMeasurements`.

### A.6 Catalog calibration details

[`determine_zero_point()`](../jpod/photometric_calibration.py#L44) first
checks the parsed header zero-point values. It returns immediately for `ZP`,
then `MAGZERO`, then `PHOTZP`, if present. Therefore it does not open ASTAP
files when any of these header values is accepted.

When there is no header value,
[`is_v_filter()`](../jpod/photometric_calibration.py#L248) checks whether
`FILTER` names one of the currently accepted V forms. It then verifies
exposure duration, celestial WCS, measurement data, catalog directory, and at
least five candidate image stars. Catalog directory resolution is performed
by [`resolve_v50_catalog_dir()`](../jpod/config.py#L30): prefer the `astap/`
folder beside the installed/source package; if absent, try `astap/` in the
selected work directory.

The cell-selection and binary-decoding steps are:

1. [`_catalog_files_for_stars()`](../jpod/photometric_calibration.py#L256)
   converts a pixel matching radius to an angular radius with the image pixel
   scale. The radius is the larger of 2 pixels and the selected FWHM. Around
   each image-star coordinate it probes the coordinate itself and eight
   directions at that radius. If, for example, these probes touch four
   distinct sky cells, those four cell files are selected (duplicate cells
   are only selected once). This finds likely cells without opening all 1,476.
2. [`_cell_for_coordinate()`](../jpod/photometric_calibration.py#L280)
   computes ASTAP's one-based declination ring and right-ascension cell. The
   filenames have the form `v50_<ring><cell>.1476`.
3. [`_read_v50_files()`](../jpod/photometric_calibration.py#L307) reads each
   selected file as bytes. It skips the 110-byte file prefix, then decodes
   **all star records in each selected cell file**, not just records already
   known to lie inside the image. Six-byte records are interpreted in order:
   special marker records provide a magnitude group and high declination
   bits; following star records encode RA and the remaining declination
   bits. The decoder returns numeric arrays in degrees and Johnson V
   magnitudes. Malformed or unreadable cells produce warnings and are skipped.
4. `SkyCoord` creates coordinates for the decoded stars;
   `image_info.wcs.world_to_pixel()` projects their positions onto the image.
   The code then discards projected positions outside the image plus a
   matching-radius border. So, for the four-file example, it decodes all
   records in those four files, but only nearby projected catalog stars go
   forward to matching.
5. [`_match_stars()`](../jpod/photometric_calibration.py#L360) uses a KD-tree
   to find, for each projected catalog star, its nearest detected image star.
   A pair is a candidate only if the two positions are within the pixel
   radius. Candidates are sorted from smallest separation to largest; the
   code accepts a candidate only if neither star has already been used.
   For example, if two catalog stars both point to the same image detection,
   the closer pair is kept and the other catalog star is left unmatched. This
   prevents duplicate assignments, though this greedy rule is not guaranteed
   to find the globally best set of pairs in crowded fields.
6. The code calculates one rate zero point per pair. It removes values more
   than three scaled MADs from the median (when the MAD is nonzero), requires
   at least five remaining values, and returns their median plus their scaled
   MAD scatter.

The algorithm is fast because it limits both the number of measured image
stars and the number of catalog cells. Its accuracy still depends on correct
WCS, usable star profiles, valid catalog magnitudes, filter agreement, and
the aperture/flux convention.

### A.7 Turning measurements into `V_lim`

[`calculate_limiting_magnitude()`](../jpod/photometric_calibration.py#L184)
checks that exposure time, zero point, SNR, FWHM, recognized image units, and
positive finite background are available. The catalog-derived zero point
uses the native image-unit flux per second: integrated-image aperture flux is
divided by exposure time, while a rate image's aperture flux is already a
rate. Header zero points retain the same per-second-flux interpretation.

The median of the background map \(B\) is converted to a per-second
background \(b\) per pixel: \(b=B/t_{\rm exp}\) for integrated images, and
\(b=B\) for images whose BUNIT is already a rate. Let \(q\) be electrons per
native image unit: the header gain for ADU/DN, and 1 for electrons. The
limiting-magnitude calculation is:

\[
A_{\rm pix}\approx\pi\,\mathrm{FWHM}^{2},\qquad
\sigma_{\rm sky,rate}^{2}
=\frac{A_{\rm pix}\,b}{q\,t_{\rm exp}}.
\]

\[
V_{\rm lim}
=ZP-2.5\log_{10}
\left(\mathrm{SNR}\,\sigma_{\rm sky,rate}\right).
\]

This yields the same rate-based limit for the four supported cases:
integrated ADU/DN, integrated electrons, ADU/DN per second, and electrons per
second. ADU/DN requires a positive `EGAIN` or `GAIN`; electron data do not
apply gain. Generic counts, missing unit clues, or conflicting clues make
automatic catalog calibration and \(V_{\rm lim}\) unavailable. The pipeline
continues with the user-provided `--vlim`. It also does not include read
noise, dark noise, source shot noise, aperture correction, or
background-estimation uncertainty. Those are explicit limitations, not
hidden corrections.

Finally, `pipeline()` passes the calculated `v_lim` rather than `args.vlim` to
[`add_detection_probability()`](../jpod/detection.py#L68) and the plot
functions. That function computes \(P_{\rm mag}\) from the asteroid's
trail-adjusted predicted V and the selected image limit. The current model
treats the limit as exact; uncertainty propagation is future work.
