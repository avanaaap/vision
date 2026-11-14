"""Prepare local data from the team's Google Drive export ZIPs.

Only local files are written. The output directory is ignored by Git.
"""

import argparse
import shutil
import zipfile
from pathlib import Path

CSV_NAMES = {
    "chexpert_train.csv",
    "chexpert_validation.csv",
    "chexpert_test.csv",
    "chexpert_official_validation.csv",
    "nih_adaptation.csv",
    "nih_test.csv",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downloads", type=Path, required=True, help="Directory with drive-download-*.zip files")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument("--nih-images", action="store_true", help="Also extract the 6,000 NIH PNGs (about 2.4 GB)")
    args = parser.parse_args()
    parts = sorted(args.downloads.glob("drive-download-*.zip"))
    if not parts:
        parser.error("No drive-download-*.zip files found")
    args.output.mkdir(parents=True, exist_ok=True)
    found_csv = set()
    image_count = 0
    for part in parts:
        with zipfile.ZipFile(part) as archive:
            for member in archive.infolist():
                name = member.filename
                if name in CSV_NAMES:
                    target = args.output / name
                    with archive.open(member) as src, target.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
                    found_csv.add(name)
                elif args.nih_images and name.startswith("nih_images/") and name.endswith(".png"):
                    target = args.output / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(member) as src, target.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
                    image_count += 1
    missing = CSV_NAMES - found_csv
    if missing:
        raise SystemExit(f"Missing split CSVs: {sorted(missing)}")
    if args.nih_images and image_count != 6000:
        raise SystemExit(f"Expected 6000 NIH images; extracted {image_count}")
    print(f"Prepared {len(found_csv)} CSVs in {args.output}")
    if args.nih_images:
        print(f"Prepared {image_count} NIH images")
    else:
        print("NIH images were not extracted. Add --nih-images before NIH evaluation.")


if __name__ == "__main__":
    main()
