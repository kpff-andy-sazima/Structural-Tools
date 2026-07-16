from pathlib import Path
from PIL import Image
import argparse
import shutil
import numpy as np


def trim_image_inplace(path: Path, threshold: int = 250, backup: bool = False) -> bool:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img)

    # non-white pixels
    mask = np.any(arr < threshold, axis=2)

    # if nothing detected
    if not mask.any():
        return False

    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]

    if len(rows) == 0 or len(cols) == 0:
        return False

    y_min, y_max = rows[0], rows[-1]
    x_min, x_max = cols[0], cols[-1]

    # no-op check
    if y_min == 0 and y_max == arr.shape[0] - 1 and x_min == 0 and x_max == arr.shape[1] - 1:
        return False

    if backup:
        shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))

    img.crop((x_min, y_min, x_max + 1, y_max + 1)).save(path)
    return True


def trim_directory(dir_path: str, threshold: int = 245, recursive: bool = False, backup: bool = False):
    root = Path(dir_path)

    files = root.rglob("*.png") if recursive else root.glob("*.png")

    total = 0
    changed = 0

    for file in files:
        total += 1
        try:
            did_crop = trim_image_inplace(file, threshold, backup=backup)
            if did_crop:
                changed += 1
                print(f"Cropped: {file}")
                if backup:
                    print(f"  backup -> {file.with_suffix(file.suffix + '.bak')}")
        except Exception as e:
            print(f"Failed: {file} -> {e}")

    print(f"\nDone. Processed {total} images, cropped {changed}.")


def main():
    parser = argparse.ArgumentParser(description="Trim whitespace from PNG images in a directory.")

    parser.add_argument("--directory", type=str, default="images", help="Directory containing PNG images")
    parser.add_argument(
        "--threshold", type=int, default=245, help="Whiteness threshold (0–255). Higher = more aggressive cropping"
    )

    parser.add_argument("--recursive", action="store_true", help="Process subdirectories recursively")

    parser.add_argument("--backup", action="store_true", help="Create .bak copies before modifying images")

    args = parser.parse_args()

    trim_directory(args.directory, threshold=args.threshold, recursive=args.recursive, backup=args.backup)


if __name__ == "__main__":
    main()
