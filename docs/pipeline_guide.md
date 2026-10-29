# J-POD pipeline guide

This guide follows the order in which one FITS image is processed by
`jpod.main.pipeline`. When a directory is selected, `main` calls this pipeline
once for each FITS file.

## 1. Load the FITS image and resolve metadata

`pipeline` starts by calling `load_image` in `image_geometry.py`. It opens the
FITS file, reads the primary image and header, builds the World Coordinate
System (WCS), and calculates the sky coordinates at the image centre and
corners. The centre is used for provider searches; the WCS is later used to
project object coordinates onto image pixels.

The loader also resolves the observatory code by calling
`get_observatory_code` in `fits_header/fits_header.py`. The priority is:

1. A valid observatory code in the FITS header.
2. A unique code inferred from the header's telescope/site name or location
   using the MPC observatory-code list.
3. The user's `-obs` / `--observatory` value, if neither previous source found
   a code.
4. Geocentric code `500`, with a warning.

The loader reads observation time from the FITS header as well. It interprets
the header time according to its keywords/comments or configured overrides,
uses exposure duration where available, and records the image start and end
times. The selected observatory code and time range are printed for the user.

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

