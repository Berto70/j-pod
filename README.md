# J-POD

## Installation
To install the `j-pod` pipeline and make the `j-pod` command available globally in your terminal, follow these steps:

### 1. Clone the repo
First, clone the project repository:
```bash
git clone https://github.com/Berto70/j-pod.git
```
### 2. Navigate to the project directory
```bash
cd j-pod
```
### 3. Install the package
Install the package in editable mode using `pip`. The `-e` flag ensures that any changes you make to the source code are immediately reflected without needing to reinstall:

(*Note: It is highly recommended to run this inside a Python virtual environment*)
```bash
pip install -e .
```
### 4. Verify the installation
Once the installation is complete, you can verify that the CLI command is registered and working by checking the help menu:
```bash
j-pod --help
```

### 5. Download the ASTAP V50 photometry catalog (optional)

J-POD uses the ASTAP **V50 photometry catalog** to calibrate V-band images
when the FITS header does not contain a usable `ZP`, `MAGZERO`, or `PHOTZP`
keyword. The V50 catalog provides Johnson-V magnitudes. The D50 database is
for astrometric plate solving and is not a substitute for V50 in J-POD's
photometric calibration.

Download V50 from the [official ASTAP download page](https://www.hnsky.org/astap.htm#index)
(look for the V50 photometry star database). Platform-specific installers
are listed there. The page also links to a
[V50 zipped archive](https://drive.google.com/file/d/13UnLqhp3GHfxrqmQ_BpnLB9CXL8VVrwX/view?usp=sharing).

J-POD does not install or download this catalog automatically. Extract or
copy the V50 sky-cell files (named `v50_<ring><cell>.1476`) into an `astap/`
directory at the repository root, beside `jpod/`. This large catalog is
ignored by Git and must be downloaded separately on each computer. The code
also checks for an `astap/` directory in the selected working directory.

Without V50, J-POD still runs. A usable zero point from the image header can
still be used; otherwise catalog calibration and the automatic limiting
magnitude are unavailable, and the pipeline falls back to `--vlim`.

### 6. Have fun with the pipeline
To use the pipeline you can proceed in two ways:
1. Move to your project/working directory and run the command from there. The package will use the current directory as root directory for the `data` and `output` directories.
2. You can run the code wherever you want. Just remember to specify the project/working directory using `-w`/`--workdir` argument. For the other directories, just specify the relative path.

**Note:** The default directory for the FITS files is `./data`. The default directory for the output files is `./output`; if `output` doesn't exist, it will be created automatically.

### Choose the small-body provider
The pipeline uses SkyBoT by default. To use MPC MPChecker instead, pass:
```bash
j-pod --provider mpc
```
Both providers'
positions and tangent-plane rates are normalized to degrees and arcseconds per
hour, respectively, before Horizons enrichment. Provider values and same-epoch
Horizons comparison values are retained separately in the in-memory table;
the saved CSV contains only the compact result columns.
See the [pipeline guide](./docs/pipeline_guide.md) for the processing sequence,
observatory-code selection, provider behavior, and output columns.

### 7. Update the package
Navigate to the project directory. Pull the new/updated files from the remote repository (be sure that you don't have pending edits). Install again the package to update the version.
```bash
cd root_path\j-pod
git pull
pip install -e .
```
