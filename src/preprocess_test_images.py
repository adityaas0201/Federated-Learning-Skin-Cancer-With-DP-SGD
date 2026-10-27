"""
Preprocess ISIC 2019 test images into a separate output folder.

Pipeline:
    1. Hair removal
    2. Border / vignette cropping
    3. CLAHE enhancement
    4. Resize to 224 x 224

Original images are never modified.
Original filenames are preserved.
"""

from pathlib import Path
import time

import cv2
import numpy as np
from tqdm import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_DIR = BASE_DIR / "ISIC_2019_Test_Input"
OUTPUT_DIR = BASE_DIR / "ISIC_2019_Test_Input_Preprocessed"

IMAGE_SIZE = 224
JPEG_QUALITY = 95

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg"}


# ============================================================
# STEP 1: HAIR REMOVAL
# ============================================================

def remove_hair(image_bgr: np.ndarray) -> np.ndarray:
    """Detect dark, thin hair-like structures and inpaint them."""

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_CROSS,
        (9, 9)
    )

    blackhat = cv2.morphologyEx(
        gray,
        cv2.MORPH_BLACKHAT,
        kernel
    )

    _, hair_mask = cv2.threshold(
        blackhat,
        10,
        255,
        cv2.THRESH_BINARY
    )

    result = cv2.inpaint(
        image_bgr,
        hair_mask,
        inpaintRadius=3,
        flags=cv2.INPAINT_TELEA
    )

    return result


# ============================================================
# STEP 2: BORDER / VIGNETTE CROPPING
# ============================================================

def crop_border(
    image_bgr: np.ndarray,
    black_thresh: int = 15
) -> np.ndarray:
    """Crop detected black borders while retaining image content."""

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    _, mask = cv2.threshold(
        gray,
        black_thresh,
        255,
        cv2.THRESH_BINARY
    )

    coords = cv2.findNonZero(mask)

    if coords is None:
        return image_bgr

    x, y, w, h = cv2.boundingRect(coords)

    height, width = image_bgr.shape[:2]

    # Avoid an excessive crop if the detected region is too small.
    if w < 0.6 * width or h < 0.6 * height:
        margin_w = int(0.05 * width)
        margin_h = int(0.05 * height)

        return image_bgr[
            margin_h:height - margin_h,
            margin_w:width - margin_w
        ]

    return image_bgr[y:y + h, x:x + w]


# ============================================================
# STEP 3: CLAHE CONTRAST ENHANCEMENT
# ============================================================

def apply_clahe(
    image_bgr: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple = (8, 8)
) -> np.ndarray:
    """Apply CLAHE to the L channel in LAB color space."""

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)

    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )

    enhanced_l = clahe.apply(l_channel)

    enhanced_lab = cv2.merge(
        (enhanced_l, a_channel, b_channel)
    )

    return cv2.cvtColor(
        enhanced_lab,
        cv2.COLOR_LAB2BGR
    )


# ============================================================
# STEP 4: COMPLETE PREPROCESSING PIPELINE
# ============================================================

def preprocess_image(
    filepath: Path,
    target_size: int = IMAGE_SIZE
) -> np.ndarray:
    """Load and preprocess one image."""

    image = cv2.imread(
        str(filepath),
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise ValueError(f"Could not read image: {filepath}")

    # 1. Hair removal
    image = remove_hair(image)

    # 2. Border cropping
    image = crop_border(image)

    # 3. CLAHE enhancement
    image = apply_clahe(image)

    # 4. Resize
    image = cv2.resize(
        image,
        (target_size, target_size),
        interpolation=cv2.INTER_AREA
    )

    return image


# ============================================================
# STEP 5: PROCESS ALL JPG/JPEG IMAGES
# ============================================================

def main():

    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f"Input folder not found:\n{INPUT_DIR}\n\n"
            "Check that ISIC_2019_Test_Input is inside "
            "the FL SKIN CANCER folder."
        )

    if not INPUT_DIR.is_dir():
        raise NotADirectoryError(
            f"Input path is not a directory: {INPUT_DIR}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    image_paths = sorted(
        path for path in INPUT_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not image_paths:
        print(f"No JPG/JPEG images found in: {INPUT_DIR}")
        return

    print("=" * 65)
    print("ISIC 2019 IMAGE PREPROCESSING")
    print("=" * 65)
    print(f"Project directory : {BASE_DIR}")
    print(f"Input directory   : {INPUT_DIR}")
    print(f"Output directory  : {OUTPUT_DIR}")
    print(f"Images found      : {len(image_paths)}")
    print(f"Output dimensions : {IMAGE_SIZE} x {IMAGE_SIZE}")
    print("=" * 65)

    start_time = time.perf_counter()

    successful = 0
    failed = []

    for image_path in tqdm(
        image_paths,
        desc="Preprocessing images"
    ):
        try:
            processed = preprocess_image(
                image_path,
                target_size=IMAGE_SIZE
            )

            # Preserve the original filename.
            output_path = OUTPUT_DIR / image_path.name

            written = cv2.imwrite(
                str(output_path),
                processed,
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
            )

            if not written:
                raise IOError(
                    f"Failed to save image: {output_path}"
                )

            successful += 1

        except Exception as exc:
            failed.append((image_path.name, str(exc)))
            tqdm.write(
                f"[ERROR] {image_path.name}: {exc}"
            )

    elapsed = time.perf_counter() - start_time

    print("\n" + "=" * 65)
    print("PREPROCESSING SUMMARY")
    print("=" * 65)
    print(f"Total images : {len(image_paths)}")
    print(f"Successful   : {successful}")
    print(f"Failed       : {len(failed)}")
    print(f"Time taken   : {elapsed:.2f} seconds")

    if elapsed > 0:
        print(
            f"Images/second: {successful / elapsed:.2f}"
        )

    print(f"Output folder: {OUTPUT_DIR}")

    if failed:
        print("\nFailed images:")
        for filename, error in failed:
            print(f"  {filename}: {error}")

    print("=" * 65)


if __name__ == "__main__":
    main()