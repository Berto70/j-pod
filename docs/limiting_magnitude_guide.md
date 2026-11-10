---
title: "Understanding Image Limiting Magnitude"
subtitle: "A beginner's guide to CCD images, photometry, and the J-POD implementation"
date: "9 November 2026"
toc-title: "Contents"
---

\newpage

# How to use this guide {.unnumbered}

This guide explains how an astronomical image can be assigned a limiting
magnitude, how J-POD currently estimates one for V-band images, and what must
be verified before that estimate can be considered robust across many images.
It is intended as a reference: terms are introduced before they are used, and
the current implementation is distinguished from future work.

The short version is:

> A limiting magnitude is not a property written into an image by nature. It
> is an estimate based on a stated filter, calibration, noise model, aperture,
> and detection threshold.

J-POD's current automatic estimate is an **image-wide, point-source, V-band
estimate using sky Poisson noise only**. It is an initial model, not yet a
validated measurement of completeness. Results should be treated cautiously
until image units, zero-point conventions, calibration scatter, and detection
performance have been checked.

## Status labels used here {.unnumbered}

- **Implemented** means the behavior exists in the current J-POD source.
- **Needs verification** means code exists, but an assumption or accuracy has
  not yet been established across different data sets.
- **Planned** means it is a recommended next development step, not current
  behavior.

\newpage

# The basic idea

Astronomers often express the brightness of a source using a **magnitude**.
Magnitudes run in the opposite direction from everyday brightness numbers:
smaller magnitude means brighter light. A difference of five magnitudes means
a factor of 100 in brightness; one magnitude is a factor of about 2.512.

An image's **limiting magnitude**, written here as \(V_{\rm lim}\) for the V
filter, is the approximate magnitude of the faintest point-like source that
the image could detect under a specified rule. For example, an SNR-3 limit
means the limiting source is estimated to produce a signal three times the
estimated noise in the chosen measurement aperture.

This does not mean every source brighter than that value is guaranteed to be
found, or every fainter source is guaranteed to be missed. Real detection
depends on the detector, image processing, source shape, background, and
detection software. A calculated limit is best understood as a model-based
threshold; injection-and-recovery tests are a stronger way to measure actual
detection completeness.

## Why J-POD needs a limit

J-POD compares predicted asteroid magnitudes with an image's sensitivity. Its
magnitude probability is currently calculated as

\[
P_{\rm mag} =
\Phi\!\left(\frac{V_{\rm lim}-V_{\rm trailing}}{\sigma_V}\right),
\]

where:

- \(V_{\rm trailing}\) is the asteroid's predicted V magnitude adjusted for
  estimated trailing loss;
- \(V_{\rm lim}\) is the image-wide limit;
- \(\sigma_V\) is the adopted uncertainty in the asteroid magnitude; and
- \(\Phi\) is the cumulative distribution function of a normal distribution.

This gives one probability based on magnitude. J-POD also calculates a
positional probability and combines the two. At present, \(V_{\rm lim}\) is
treated as exact in this probability calculation. Uncertainty in the limit is
not yet propagated into \(P_{\rm mag}\).

# What is inside a CCD image?

## Light becomes charge, then numbers

A digital astronomical camera has light-sensitive detector elements called
**pixels**. A photon that is absorbed by the detector can release an electron.
The detector collects these electrons as electrical charge during an
exposure. Electronics then measure the charge and an analog-to-digital
converter turns the measurement into a number stored in the image.

The chain is approximately:

```text
photons -> detector electrons -> measured voltage/charge -> digital pixel value
```

The number in a FITS image is therefore not automatically a photon count or
an electron count. It may be a digital number, a calibrated electron value, a
rate, or a processed value. FITS headers sometimes describe the units, but
headers are not always complete or consistent.

## Electrons, ADU, DN, and gain

- **Electron (e-)**: a unit of detector charge. Photon arrival is statistical;
  for \(N\) detected photons/electrons, the Poisson standard deviation is
  approximately \(\sqrt{N}\) electrons.
- **ADU**: analog-to-digital unit, a digital number representing the detector
  measurement.
- **DN**: digital number. In many image files it is used much like ADU.
- **Gain**: a conversion factor between image units and electrons. A common
  definition is \(g = \mathrm{electrons}/\mathrm{ADU}\). A gain of
  \(0.83\ \mathrm{e^-}/\mathrm{ADU}\) means one ADU corresponds on average to
  0.83 collected electrons.
- **BUNIT**: a FITS header keyword that may state the units of pixel values.
  It can be absent, generic, or insufficient to describe whether values are
  integrated over the exposure or already divided by exposure time.

If pixel values are in ADU, Poisson noise is not generally
\(\sqrt{\text{ADU}}\) unless the gain is exactly one electron per ADU. If
pixel values are in electrons, gain conversion is not needed for the
Poisson-noise calculation. If values are rates, the exposure-time conversion
must be handled differently.

## Other detector and image effects

These are common terms in CCD/CMOS imaging. They are **not all included** in
J-POD's current limiting-magnitude estimate.

- **Bias**: an electronic baseline added to avoid negative digitized values.
- **Dark current**: electrons produced by the detector even without incoming
  light. It generally increases with exposure time and temperature.
- **Read noise**: uncertainty added when the detector electronics read each
  pixel.
- **Flat field**: a correction for pixel-to-pixel sensitivity differences and
  illumination effects such as vignetting.
- **Stacking**: combining multiple exposures. Noise depends on how they were
  combined, rejected, rescaled, and weighted.
- **Saturation**: the detector or digitizer reaches its maximum representable
  value. Saturated stars cannot provide reliable flux measurements.

Images may already have had bias, dark, and flat corrections applied.
Reduction software may also rescale pixel values. Therefore, a FITS file
cannot always be interpreted from its pixel values alone; image provenance
and header documentation matter.

# Filters, bands, and magnitudes

## A filter selects colors of light

A filter passes some wavelengths and blocks others. Astronomers name
standardized wavelength ranges with band labels such as **B**, **V**, and
**R**. The V band is centered in visible light; R is redder. Other systems,
such as the Sloan survey, use labels such as \(g'\), \(r'\), and \(i'\).

The filter's full wavelength response is its **passband** or **transmission
curve**. A filter labelled `R` is not guaranteed to be identical to every
other filter labelled `R`: detector sensitivity, filter glass, optics, and
standard-system definitions can differ.

## A magnitude always belongs to a band and a system

A magnitude is measured through a passband. The same star can have different
values in V and R because its emitted light varies with wavelength. A catalog
value of \(V=15\) is not automatically an R magnitude of 15.

Magnitude systems also define how flux is referenced. Two common examples are
the Vega system and the AB system. A value in one system should not be mixed
with another without the appropriate convention or transformation.

## Color indexes and asteroid comparisons

A **color index** is a difference between magnitudes in two filters, for
example

\[
V-R.
\]

It describes how an object's measured brightness changes between those
bands. A redder object and a bluer object with the same R magnitude can have
different V magnitudes.

For an R-filter image, a defensible workflow would be:

1. Calibrate the image in its actual R-like passband using compatible
   reference-star magnitudes.
2. Compute the image's limit in that native passband, such as \(R_{\rm lim}\).
3. If comparison with asteroid V magnitudes is required, convert the one
   image-wide limit using an explicitly chosen representative asteroid color:
   \[
   V_{\rm lim} \approx R_{\rm lim} + (V-R)_{\rm assumed}.
   \]

The assumed color would be common to the image and the asteroid population
being studied, so the result remains one image-wide number. It is approximate:
individual asteroids have different colors. J-POD does **not** implement
this conversion yet.

## Current filter scope

**Implemented:** catalog-derived zero-point calibration is restricted to
images explicitly identified as V band because the local ASTAP V50 catalog is
being used for Johnson V magnitudes.

**Not implemented:** calibration in R or Sloan bands, filter transformations,
or asteroid-color corrections. Non-V images continue to use the user-provided
`--vlim` fallback. A header zero point can still be read, but its filter and
photometric convention are not independently validated.

# Star images, FWHM, and apertures

## Why a star is a spot, not one pixel

Even a point-like star is recorded across multiple pixels. The telescope,
atmosphere, focus, tracking, and detector spread its light into a small image
called the **point-spread function (PSF)**. A star's apparent width can vary
across the image and between exposures.

## FWHM

**FWHM** means full width at half maximum. Imagine a cross-section through a
star's central peak. FWHM is the width of the star at the points where its
brightness has fallen to half the peak height. It is a width, not a radius.

J-POD estimates the widths of suitable stellar profiles and uses their robust
median. In the current measurement pass it seeks compact, relatively round,
bright profiles; excludes saturated and nearby blended candidates; and uses
at most the brightest 100 profile candidates. A plausible FITS header FWHM is
preferred. If the header value differs from the measured value by more than a
factor of two, J-POD reports a note and uses the image-measured value.

## Aperture and aperture flux

An **aperture** is a region around a source whose pixels are summed to
estimate its brightness. J-POD currently uses a circular aperture with
**radius equal to one FWHM**. For FWHM \(=3.28\) pixels, this means radius
\(3.28\) pixels, not diameter \(3.28\) pixels.

The **aperture flux** is the sum of the pixel values inside that circle after
subtracting the estimated background. It is measured in the image's pixel
units. It is not the brightness of one pixel.

For a circular aperture of radius \(r=\mathrm{FWHM}\), the continuous-area
approximation is

\[
A_{\rm pix} \approx \pi\,\mathrm{FWHM}_{\rm pixels}^{\,2}.
\]

The real software sums discrete pixels whose centers fall inside the circle;
the limiting-magnitude calculation uses the continuous area approximation
above. For an ideal two-dimensional Gaussian PSF, a radius of one FWHM
encloses about 94% of the total light. Real profiles may differ.

The catalog zero point is derived using this same aperture. That makes the
zero point tied to the aperture measurement. A future validation should check
whether this aperture is appropriate and whether an aperture correction is
needed for different PSFs or image-processing pipelines.

# Background and noise

## What is the background?

The image **background** is the broad level underneath sources. It can
include sky brightness (natural airglow, moonlight, light pollution, and
scattered light), detector offset, dark signal, and residuals from image
processing. Thus "measured image background" is not necessarily pure
astronomical sky.

Background and background noise are different:

- **Background level** is the average/typical pixel value underlying sources.
- **Noise** is the random variation around that level.

A bright, smooth sky can have a high background and correspondingly high
photon noise. A low background does not guarantee low noise if read noise or
processing artifacts are large.

## How J-POD estimates it

**Implemented:** J-POD uses Photutils `Background2D` with a 50-by-50-pixel
mesh and a median estimator to make a spatial background map. It subtracts
this map to find stars and estimate the residual scatter used by star
detection.

For the limiting-magnitude formula, J-POD takes the median of the background
map. It treats this as the typical image-wide background per pixel for the
exposure and uses that value directly in the sky-noise formula. A background
rate (background divided by exposure time) can help compare images with
different exposure durations, but is not needed for this per-image
calculation. The residual-noise estimate used during star finding is
**not** the \(\sigma_{\rm sky}\) in the current limiting-magnitude equation.

Using one median creates one image-wide limit. It does not describe local
changes due to vignetting, gradients, crowded fields, nebulosity, or a bright
Moon gradient.

## Sky Poisson noise

Photons arrive randomly. If the expected number of detected sky electrons is
\(N_e\), the Poisson standard deviation is approximately \(\sqrt{N_e}\)
electrons and the variance is \(N_e\) electrons squared.

Let:

- \(A_{\rm pix}\) be the number of pixels in the measurement aperture;
- \(B_{\rm image}\) be the accumulated background in image units per pixel
  for the whole exposure;
- \(t_{\rm exp}\) be exposure time in seconds; and
- \(g\) be gain in electrons per image unit.

The current implementation uses the median full-exposure background
\(B_{\rm image}\) directly. It estimates the sky-noise variance in image units
squared as

\[
\sigma_{\rm sky,image}^{\,2}
=
A_{\rm pix}\,B_{\rm image}/g.
\]

Then

\[
\sigma_{\rm sky,image}
=
\sqrt{A_{\rm pix}\,B_{\rm image}/g}.
\]

If pixel units are electrons, use the equivalent formula with \(g=1\)
electron per image unit. If pixel units are ADU, dividing the variance by
electrons-per-ADU converts the electron shot noise back to ADU units.

**Important current assumption:** J-POD reads `EGAIN` first, then `GAIN`, and
interprets either as electrons per image unit. It does not currently inspect
`BUNIT` to determine whether pixel values are ADU, electrons, or rates. If no
gain is found, it assumes \(g=1\). These assumptions are not safe for every
FITS file and are a high-priority item to improve.

## Noise not yet included

The current limit includes sky Poisson noise only. It does not include:

- read noise from reading the detector;
- dark-current shot noise;
- source Poisson noise at the detection threshold;
- uncertainty in estimating/subtracting the background;
- correlated noise introduced by resampling or stacking;
- cosmic rays, bad pixels, or other defects.

Moonlight is not a separate noise term in this first model; its smooth
contribution may raise the measured background, but gradients may make a
single median inadequate. The chosen omissions mean the current limit is an
approximation, not a complete detector noise model.

# Zero point: connecting image numbers to magnitudes

## What a zero point does

The image records flux in instrumental units, not standard magnitudes. A
**photometric zero point** is a calibration constant that relates measured
flux to a catalog magnitude.

J-POD uses a count-rate convention:

\[
m = ZP - 2.5\log_{10}(F/t_{\rm exp}),
\]

or equivalently

\[
ZP = m + 2.5\log_{10}(F/t_{\rm exp}),
\]

where \(F\) is the background-subtracted aperture flux in image units and
\(F/t_{\rm exp}\) is its rate in image units per second. Brighter sources
have larger flux and smaller magnitudes.

This convention means the numerical zero point depends on the units and
exposure-time convention of the flux. A zero point for ADU/s is not
automatically interchangeable with a zero point for electrons/s or for
uncorrected ADU totals.

## Current zero-point procedure

For each image, the current pipeline:

1. Loads pixels, WCS, exposure, filter, and selected header metadata.
2. Measures background, FWHM, candidate star profiles, and aperture fluxes
   in one shared image-analysis pass. If there are no suitable stars and no
   usable header FWHM, it preserves the background measurements and marks
   FWHM unavailable instead of stopping.
3. Checks for a numeric `MAGZERO` header value; if present, uses it.
   Otherwise it checks `PHOTZP`. If both are present but differ, `MAGZERO`
   wins and the program warns.
4. If there is no header zero point, only proceeds with catalog calibration
   if the filter is recognized as V and the local V50 catalog is available.
5. Selects nearby V50 sky-cell files, projects catalog coordinates into image
   pixels using the FITS WCS, and matches catalog stars to measured stars.
6. Requires at least five matches, uses at most the 50 brightest measured
   stars, rejects strong outliers with a median/MAD rule, and reports the
   median of individual zero-point estimates.

If a usable header zero point exists, the code returns it without opening the
catalog. The shared image-measurement pass has already run, however, because
background measurements are needed by the pipeline.

When FWHM cannot be measured and is also absent from the header, J-POD skips
the trailing-loss correction and the automatic limiting-magnitude estimate.
It retains the unadjusted asteroid \(V\) for magnitude probabilities and
keeps the user-provided `--vlim`. The output marks `V_trailing`, the
trail-to-FWHM ratio, and trailing loss as unavailable.

## What the J75 numbers mean

The test image `data/J75_01_20110720204510.fits` has `FILTER='V'`,
`EXPTIME=90 s`, and header gain `EGAIN=0.83 electrons/ADU`. The current
processing produced approximately:

- image-measured FWHM: \(3.28\) pixels (the header says 90 pixels, so the
  image-measured value is used);
- matched catalog stars: 7;
- median catalog-derived zero point: \(21.713\);
- robust scatter of individual star zero points: \(0.603\) mag;
- median background: \(2491.7\) image units/pixel for the exposure, or
  \(27.69\) image units/pixel/second;
- approximate aperture area: \(\pi(3.28)^2 \approx 33.89\) pixels;
- sky noise in the aperture: about \(318.95\) image units; and
- calculated sky-only limit at SNR 3: about \(V_{\rm lim}=19.15\).

The 0.603-mag scatter is a robust spread among the individual matched-star
zero-point estimates. It is **not yet a confidence interval or standard error
on \(V_{\rm lim}\)**. It is relatively large and makes this J75 estimate
provisional. More and better matches, a validated photometric aperture, and
understanding of the header/calibration conventions are needed before
relying on it for quantitative probabilities.

## Header zero points need validation

The current code accepts finite `MAGZERO` and `PHOTZP` values without
independently establishing their filter, flux units, aperture convention, or
whether exposure time is already included. A header value can therefore be
numerically usable but physically incompatible with the image measurements.
Checking its plausibility against catalog stars is a useful future feature,
but that comparison must first ensure both values use the same filter and
flux convention.

# The current limiting-magnitude calculation

## Formula

With aperture radius equal to FWHM,

\[
A_{\rm pix}\approx\pi\,\mathrm{FWHM}_{\rm pixels}^{\,2}.
\]

The program estimates sky noise as

\[
\sigma_{\rm sky}
=
\sqrt{A_{\rm pix}\,B_{\rm rate}\,t_{\rm exp}/g}.
\]

It then estimates the limiting magnitude with

\[
V_{\rm lim}
=
ZP + 2.5\log_{10}\!\left(
\frac{t_{\rm exp}}{\mathrm{SNR}_{\rm threshold}\,
\sigma_{\rm sky}}
\right).
\]

Here \(ZP\) is calibrated for image units per second and
\(\sigma_{\rm sky}\) is in image units. In this shorthand, the zero point
defines the reference rate unit. In code, the equivalent operation is to
calculate the limiting aperture-flux rate,
\(\mathrm{SNR}_{\rm threshold}\,\sigma_{\rm sky}/t_{\rm exp}\), and compare
it with the calibrated rate zero point.

## SNR threshold

**SNR** is signal-to-noise ratio: source signal divided by the uncertainty
in that signal measurement. The default threshold is 3. It can be changed
with `--snr`, for example `--snr 5`. A higher threshold is more conservative
and produces a brighter (numerically smaller) limiting magnitude.

## How the limit is used

For V-band images with an available zero point and usable background, the
calculated value replaces `--vlim` for the detection-probability calculation
and plots. If the image is not recognized as V, the zero point is unavailable,
or the limiting calculation fails, J-POD falls back to the supplied
`--vlim` value (default 21.5).

The calculation produces one global point-source limit using the median
background and one representative FWHM. Asteroid trails are handled
separately by the existing trailing-loss estimate, which adjusts predicted
asteroid magnitude before computing the magnitude probability. This does not
make the image limit a local map of sensitivity.

# What has been implemented, and what has not

## Implemented now

- FITS loading collects filter, exposure, selected zero-point and FWHM
  keywords, saturation level, and gain keywords.
- One shared image-analysis pass estimates a spatial background map, a robust
  residual scatter for star detection, stellar profiles, FWHM, and aperture
  fluxes.
- Header `MAGZERO` is preferred to `PHOTZP`; if neither is present, V50
  catalog matching is attempted only for recognized V filters.
- V50 lookup reads sky cells near measured stars rather than scanning the
  whole catalog.
- Catalog-derived calibration requires at least five matches and reports
  robust scatter.
- A sky-only V-band limit is calculated using the median background,
  FWHM-radius aperture, gain if present (otherwise assumed unit gain), and
  configurable SNR threshold.
- The automatic limit is passed into magnitude-probability calculations and
  plots; `--vlim` remains a fallback.

## Explicitly not implemented

- General recognition of whether pixel values are ADU, electrons, counts per
  second, or rescaled/stacked units.
- Validation that header zero points share the filter and rate/aperture
  convention used by the calculation.
- Read-noise, dark-current, source-shot-noise, or resampling-noise terms.
- A spatial limiting-magnitude map or a validated typical/conservative
  image-wide background policy.
- Uncertainty propagation from FWHM, zero point, background, gain, or catalog
  magnitudes into a \(V_{\rm lim}\) interval.
- Propagation of limit uncertainty into a range or distribution for
  \(P_{\rm mag}\).
- Calibration for non-V filters or a justified conversion from another band
  to asteroid V magnitudes.
- Validation against measured detection completeness using artificial-source
  injection and recovery.

# Remaining work, in priority order

This is the recommended sequence for making the estimate robust across many
images. Earlier steps support the correctness of later ones.

## Priority 1 — Establish pixel units and gain

Read and interpret `BUNIT`, `EGAIN`/`GAIN`, comments, and image provenance.
Recognize integrated ADU, integrated electrons, rates, and known processed
products where possible. Convert sky Poisson variance into the same units as
the flux used for the zero point. If units cannot be determined, do not
silently assume unit gain: report the assumption or fall back safely.

## Priority 2 — Verify the zero point and J75 calibration scatter

Inspect the individual matched stars and their positions, fluxes, magnitudes,
and per-star zero points. Investigate J75's 0.603-mag robust scatter. Check
match quality, saturation, blending, background subtraction, catalog
uncertainty, filter identity, and whether the aperture/zero-point convention
is valid. Define and validate the meanings of header zero-point keywords.

## Priority 3 — Define the image-wide background and aperture rule

Decide whether the limit means typical sensitivity, a conservative value, or
another chosen statistic. Test the median background against gradients,
vignetting, bright stars, bad pixels, crowded fields, and source-free areas.
Validate whether one FWHM is the right aperture radius and how its discrete
pixel area should be calculated. Document the point-source assumption and
any aperture correction.

## Priority 4 — Propagate uncertainty into the limit and probability

Estimate uncertainty in FWHM, zero point, and background. Because FWHM affects
both the calibration aperture and the sky-noise aperture, propagate these
inputs together, for example by resampling the calibration stars and
stellar-profile measurements and recalculating \(ZP\), \(\sigma_{\rm sky}\),
and \(V_{\rm lim}\). Use the resulting limit distribution to compute a
probability interval or distribution for \(P_{\rm mag}\). Keep asteroid
magnitude uncertainty separate from image-limit uncertainty, and state which
sources are included.

## Priority 5 — Add other noise sources

After the sky-only path is validated, add read noise, dark-current noise, and
source shot noise when the image metadata supports them. Account for
correlated noise in resampled/stacked data where relevant. These terms should
be added in variance space with compatible units, not by adding standard
deviations blindly.

## Priority 6 — Expand filter support

Identify the actual filter system, use compatible reference-star magnitudes,
and calibrate limits in the native band. If asteroid V magnitudes remain the
comparison target, provide an explicit color-based conversion with a
documented representative asteroid color and uncertainty. Do not silently
treat R, Sloan \(r'\), and V as interchangeable.

## Priority 7 — Validate performance on known and synthetic cases

Use synthetic images with known backgrounds, gain, FWHM, zero point, and
source fluxes. Test real images with trustworthy photometry. Check sensible
responses to changes in exposure, background, and seeing. Finally, inject
artificial sources and measure recovery rate versus magnitude; this tests
actual detection completeness rather than only the analytic noise threshold.

### Immediate next development task

The most useful next task is **Priority 1 together with the J75 checks in
Priority 2**: establish image units and gain handling, then inspect why the
seven matched J75 stars produce a large zero-point spread. Until those checks
are done, the automatic number is a useful experiment but not a robust
cross-instrument result.

# Glossary

**ADU (analog-to-digital unit)** — A digital value representing measured
detector charge or voltage. Often similar to DN.

**Airmass** — A measure of the amount of atmosphere along a line of sight.
Larger airmass generally means more extinction. A calibrated in-field
zero point already reflects the observed atmosphere for that exposure, so an
additional correction must not be applied without checking its convention.

**Aperture** — A selected region around a source whose pixel values are summed.

**Aperture flux** — Background-subtracted sum of image values in an aperture.

**Aperture correction** — Adjustment for light outside the chosen aperture
when converting aperture flux to a total-source flux convention.

**Astrometric catalog** — A catalog giving sky positions; it may or may not
also provide suitable photometric magnitudes.

**Background** — The image level underneath sources; it may combine sky,
detector, and processing contributions.

**Band / passband** — A wavelength range selected by a filter and its
detector/optical response, such as V or R.

**Bias** — Electronic offset added to detector measurements.

**CCD / CMOS** — Common solid-state detector technologies used in digital
cameras. Both convert incoming light into measurable charge.

**Color / color index** — Difference between magnitudes in different bands,
such as \(V-R\); describes how an object's brightness changes with wavelength.

**Dark current** — Detector-generated charge accumulated without incident
light.

**DN (digital number)** — A pixel's stored numerical value; often used
interchangeably with ADU.

**Exposure time (\(t_{\rm exp}\))** — Duration for which the detector
collected light.

**FITS** — Flexible Image Transport System, a standard astronomical file
format containing arrays and metadata headers.

**FWHM** — Full width at half maximum; a measure of a star's image width.

**Gain (\(g\))** — Conversion between image units and electrons, commonly
electrons per ADU.

**Limiting magnitude (\(V_{\rm lim}\))** — Estimated faintest source magnitude
detectable under a stated SNR and measurement procedure.

**Magnitude** — Logarithmic brightness scale; a smaller value is brighter.
Magnitudes must be associated with a band and photometric system.

**MAD (median absolute deviation)** — Robust measure of spread around a
median. A scaled MAD is used in the code to reduce the influence of outliers.

**Photometry** — Measurement and calibration of source brightness.

**Photon shot noise / Poisson noise** — Random variation in photon arrivals;
for \(N\) counts the standard deviation is approximately \(\sqrt{N}\).

**PSF (point-spread function)** — Image shape produced by a point-like source
such as a distant star.

**Read noise** — Noise introduced by detector readout electronics.

**SNR (signal-to-noise ratio)** — Measured source signal divided by its
estimated uncertainty.

**Zero point (\(ZP\))** — Calibration constant relating instrumental
flux/rate to a standard magnitude for a specified filter and convention.

**WCS (World Coordinate System)** — FITS metadata mapping image pixels to sky
coordinates such as right ascension and declination.

# Further reading

These sources provide additional detail and define relevant data standards:

1. **J-POD implementation:** `jpod/image_geometry.py`,
   `jpod/image_analysis.py`, `jpod/photometric_calibration.py`,
   `jpod/detection.py`, `jpod/trailing_loss.py`, and
   `docs/pipeline_guide.md`.
2. **FITS Standard:** NASA/GSFC FITS Support Office,
   <https://fits.gsfc.nasa.gov/fits_standard.html>.
3. **Photutils background estimation:** Astropy Photutils documentation,
   <https://docs.astropy.org/projects/photutils/en/stable/user_guide/background.html>.
4. **ASTAP star databases:** ASTAP documentation and database downloads,
   <https://www.hnsky.org/astap.htm#index>.
5. **APASS multi-band photometry:** AAVSO APASS overview,
   <https://archive.aavso.org/apass>.
6. **Asteroid multi-band photometry:** NASA Planetary Data System Small
   Bodies Node, SDSS Moving Object Catalog,
   <https://sbn.psi.edu/pds/resource/sdssmoc.html>.

The code behavior described here was checked against the J-POD workspace on
9 November 2026. External catalog products and documentation can change;
check their current release notes before relying on catalog details.
