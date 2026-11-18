from astropy.io import fits
from pathlib import Path
import pandas as pd

j_var_dir = Path("~/Desktop/j-pod/data/j-var:rSDSS").expanduser()
csv_path = Path(j_var_dir / "jvar_t80cam_calibrated_images.csv")
df = pd.read_csv(csv_path)
header_map = {
    "depth3arc5s": ("MAGLIM", "5-sigma point-source limiting magnitude"),
    "zp": ("ZP", "Photometric zero point"),
    "fwhm_arcsec": ("OAJ PRO FWHMSEXT", "Full width at half maximum in arcseconds"),
    "exptime_s": ("TEXPTIME", "Exposure time in seconds"),
}
for _, row in df.iterrows():
    filename =f"{row["namered"]}.fz"
    image_path = j_var_dir / filename
    if not image_path.exists():
        print(f"File {image_path} does not exist. Skipping.")
        continue

    with fits.open(image_path, mode="update") as hdul:
        header = hdul["IMAGE"].header
        for col_name, (fits_key, comment) in header_map.items():
            val = row[col_name]
            # Skip NaN/missing values
            if pd.notna(val):
                header[fits_key]=(val, comment)
    print(f"Updated header for {image_path}")
