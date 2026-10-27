"""
01b_preprocess_images.py
------------------------
PHASE 1 - STEP 1b

RUNS AFTER:
    01_prepare_data.py

RUNS BEFORE:
    02_dirichlet_partition.py

WHAT THIS DOES:
    Applies three image-level preprocessing steps to every image in the
    train / validation / test splits.

    1. HAIR REMOVAL
       DullRazor-style black-hat morphology + inpainting.

    2. BORDER / VIGNETTE CROPPING
       Removes black borders and irrelevant edge regions.

    3. CLAHE
       Enhances local contrast using the LAB color space.

    4. RESIZE
       Converts every image to IMAGE_SIZE x IMAGE_SIZE.

WHY THIS ORDER:
    Hair removal -> Border crop -> CLAHE -> Resize

    Hair removal is performed before contrast enhancement because CLAHE
    could make dark hair strands more prominent.

    Cropping is performed before CLAHE so that contrast enhancement is
    applied only to the useful image region.

IMPORTANT RERUN SAFETY:
    The original train/val/test CSV paths are saved as backup files.

    If this script is executed again, it ALWAYS reads the raw-path
    backup instead of the already-processed CSV.

    Therefore:

        RAW IMAGE
            |
            v
        preprocessing
            |
            v
        processed image

    and never:

        processed image
            |
            v
        preprocessing again

TIMING:
    The script reports:

        - time for each split
        - average time per image
        - images processed per second
        - total preprocessing time

OUTPUT:
    outputs/
    ├── processed_images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    │
    ├── train_split.csv
    ├── val_split.csv
    ├── test_split.csv
    │
    ├── train_split_raw_paths_backup.csv
    ├── val_split_raw_paths_backup.csv
    └── test_split_raw_paths_backup.csv
"""

import os
import sys
import time

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


# =====================================================================
# IMPORT CONFIG
# =====================================================================

# Project structure:
#
# fl_project/
# ├── configs/
# │   └── config.py
# ├── src/
# │   └── 01b_preprocess_images.py
# ├── outputs/
# └── HAM10000/
#
# Therefore:
#
# src/../configs/config.py
#
# is the correct location.

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "configs"
    )
)

import config


# =====================================================================
# STEP A: HAIR REMOVAL
# =====================================================================

def remove_hair(image_bgr: np.ndarray) -> np.ndarray:
    """
    Removes hair strands using a DullRazor-style approach.

    Steps:
        1. Convert image to grayscale.
        2. Apply morphological black-hat filtering.
        3. Threshold dark line-like structures.
        4. Inpaint detected hair pixels.

    Parameters
    ----------
    image_bgr : np.ndarray
        Input image in OpenCV BGR format.

    Returns
    -------
    np.ndarray
        Hair-reduced BGR image.
    """

    # Convert BGR image to grayscale.
    gray = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2GRAY
    )

    # Cross-shaped kernel detects thin dark structures.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_CROSS,
        (9, 9)
    )

    # Black-hat operation highlights dark structures
    # against a brighter background.
    blackhat = cv2.morphologyEx(
        gray,
        cv2.MORPH_BLACKHAT,
        kernel
    )

    # Threshold dark structures to create hair mask.
    _, hair_mask = cv2.threshold(
        blackhat,
        10,
        255,
        cv2.THRESH_BINARY
    )

    # Fill detected hair pixels using surrounding image information.
    inpainted = cv2.inpaint(
        image_bgr,
        hair_mask,
        inpaintRadius=3,
        flags=cv2.INPAINT_TELEA
    )

    return inpainted


# =====================================================================
# STEP B: BORDER / VIGNETTE CROPPING
# =====================================================================

def crop_border(
    image_bgr: np.ndarray,
    black_thresh: int = 15
) -> np.ndarray:
    """
    Removes black vignette borders.

    A threshold identifies pixels that are not part of the black border.
    The bounding box of those pixels is used as the crop.

    If the detected region is suspiciously small, a safe 90% center crop
    is used instead.
    """

    # Convert to grayscale.
    gray = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2GRAY
    )

    # Find pixels brighter than the black-border threshold.
    _, mask = cv2.threshold(
        gray,
        black_thresh,
        255,
        cv2.THRESH_BINARY
    )

    # Find coordinates of non-black pixels.
    coords = cv2.findNonZero(mask)

    # If nothing was detected, keep the original image.
    if coords is None:
        return image_bgr

    # Calculate bounding rectangle.
    x, y, w, h = cv2.boundingRect(coords)

    h_img, w_img = image_bgr.shape[:2]

    # Safety check.
    #
    # If the detected region is smaller than 60% of the image,
    # the threshold probably detected something incorrectly.
    if w < 0.6 * w_img or h < 0.6 * h_img:

        # Safe 5% margin from each side.
        margin_w = int(0.05 * w_img)
        margin_h = int(0.05 * h_img)

        return image_bgr[
            margin_h:h_img - margin_h,
            margin_w:w_img - margin_w
        ]

    # Normal border crop.
    return image_bgr[
        y:y + h,
        x:x + w
    ]


# =====================================================================
# STEP C: CLAHE
# =====================================================================

def apply_clahe(
    image_bgr: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple = (8, 8)
) -> np.ndarray:
    """
    Applies CLAHE to the L channel of LAB color space.

    Using only the L channel improves local contrast while preserving
    the original color information in the A and B channels.
    """

    # Convert BGR -> LAB.
    lab = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2LAB
    )

    # Split LAB channels.
    l_channel, a_channel, b_channel = cv2.split(lab)

    # Create CLAHE object.
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )

    # Enhance local contrast.
    l_enhanced = clahe.apply(
        l_channel
    )

    # Recombine channels.
    enhanced_lab = cv2.merge(
        (
            l_enhanced,
            a_channel,
            b_channel
        )
    )

    # Convert LAB -> BGR.
    enhanced_bgr = cv2.cvtColor(
        enhanced_lab,
        cv2.COLOR_LAB2BGR
    )

    return enhanced_bgr


# =====================================================================
# FULL IMAGE PREPROCESSING PIPELINE
# =====================================================================

def preprocess_image(
    filepath: str,
    target_size: int = config.IMAGE_SIZE
) -> np.ndarray:
    """
    Complete preprocessing pipeline:

        Load
          ↓
        Hair removal
          ↓
        Border crop
          ↓
        CLAHE
          ↓
        Resize
          ↓
        Return processed image

    Returns
    -------
    np.ndarray
        Processed BGR image.
    """

    # Load image.
    image = cv2.imread(filepath)

    if image is None:
        raise ValueError(
            f"Could not read image: {filepath}"
        )

    # -------------------------------------------------------------
    # Step 1: Hair removal
    # -------------------------------------------------------------

    image = remove_hair(image)

    # -------------------------------------------------------------
    # Step 2: Border / vignette crop
    # -------------------------------------------------------------

    image = crop_border(image)

    # -------------------------------------------------------------
    # Step 3: CLAHE
    # -------------------------------------------------------------

    image = apply_clahe(image)

    # -------------------------------------------------------------
    # Step 4: Resize
    # -------------------------------------------------------------

    image = cv2.resize(
        image,
        (
            target_size,
            target_size
        ),
        interpolation=cv2.INTER_AREA
    )

    return image


# =====================================================================
# PROCESS ONE SPLIT
# =====================================================================

def process_split(
    split_name: str,
    processed_root: str
):
    """
    Processes one dataset split.

    Example:
        split_name = "train"

    The function:

        1. Finds train_split.csv.
        2. Creates/uses raw-path backup.
        3. Processes raw images.
        4. Saves processed images.
        5. Updates train_split.csv.
        6. Reports timing statistics.
    """

    # -------------------------------------------------------------
    # Paths
    # -------------------------------------------------------------

    csv_path = os.path.join(
        config.OUTPUT_DIR,
        f"{split_name}_split.csv"
    )

    backup_path = os.path.join(
        config.OUTPUT_DIR,
        f"{split_name}_split_raw_paths_backup.csv"
    )

    split_out_dir = os.path.join(
        processed_root,
        split_name
    )

    # -------------------------------------------------------------
    # Check main CSV
    # -------------------------------------------------------------

    if not os.path.exists(csv_path):

        raise FileNotFoundError(
            f"Could not find {csv_path}.\n"
            "Run 01_prepare_data.py first."
        )

    # -------------------------------------------------------------
    # IMPORTANT:
    # Always use the RAW path backup if available.
    #
    # This prevents preprocessing processed images again.
    # -------------------------------------------------------------

    if os.path.exists(backup_path):

        print(
            f"\nUsing raw-path backup:"
            f"\n  {backup_path}"
        )

        df = pd.read_csv(
            backup_path
        )

    else:

        print(
            f"\nCreating raw-path backup:"
            f"\n  {backup_path}"
        )

        # First execution:
        # train_split.csv still contains raw image paths.
        df = pd.read_csv(
            csv_path
        )

        # Save untouched raw paths.
        df.to_csv(
            backup_path,
            index=False
        )

    # -------------------------------------------------------------
    # Create output directory
    # -------------------------------------------------------------

    os.makedirs(
        split_out_dir,
        exist_ok=True
    )

    # -------------------------------------------------------------
    # Counters
    # -------------------------------------------------------------

    new_paths = []

    failed = 0

    successful = 0

    # -------------------------------------------------------------
    # Start timer
    # -------------------------------------------------------------

    split_start_time = time.perf_counter()

    print(
        f"\nProcessing {split_name} split: "
        f"{len(df)} images"
    )

    print(
        f"Output directory: "
        f"{split_out_dir}"
    )

    # -------------------------------------------------------------
    # Process every image
    # -------------------------------------------------------------

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc=split_name
    ):

        try:

            # Read RAW image path.
            raw_filepath = row["filepath"]

            # Run preprocessing pipeline.
            processed = preprocess_image(
                raw_filepath
            )

            # Output path.
            out_path = os.path.join(
                split_out_dir,
                f"{row['image_id']}.jpg"
            )

            # Save processed image.
            write_success = cv2.imwrite(
                out_path,
                processed
            )

            # Explicitly verify that OpenCV actually wrote the file.
            if not write_success:

                raise IOError(
                    f"cv2.imwrite failed: {out_path}"
                )

            # Verify file exists.
            if not os.path.exists(out_path):

                raise IOError(
                    f"Output file was not created: "
                    f"{out_path}"
                )

            new_paths.append(
                out_path
            )

            successful += 1

        except Exception as e:

            print(
                f"\n[WARNING] Failed on "
                f"{row['image_id']}: {e}"
            )

            # Fall back to the original RAW path.
            new_paths.append(
                row["filepath"]
            )

            failed += 1

    # -------------------------------------------------------------
    # Calculate timing
    # -------------------------------------------------------------

    split_elapsed = (
        time.perf_counter()
        - split_start_time
    )

    total_images = len(df)

    avg_time = (
        split_elapsed / total_images
        if total_images > 0
        else 0
    )

    images_per_second = (
        total_images / split_elapsed
        if split_elapsed > 0
        else 0
    )

    # -------------------------------------------------------------
    # Update dataframe
    # -------------------------------------------------------------

    df["filepath"] = new_paths

    # -------------------------------------------------------------
    # Save updated CSV
    # -------------------------------------------------------------

    df.to_csv(
        csv_path,
        index=False
    )

    # -------------------------------------------------------------
    # Print results
    # -------------------------------------------------------------

    print(
        f"\n{'-' * 60}"
    )

    print(
        f"{split_name.upper()} PREPROCESSING COMPLETE"
    )

    print(
        f"{'-' * 60}"
    )

    print(
        f"Total images       : {total_images}"
    )

    print(
        f"Successful         : {successful}"
    )

    print(
        f"Failed             : {failed}"
    )

    print(
        f"Time taken         : {split_elapsed:.2f} seconds"
    )

    print(
        f"Time taken         : "
        f"{split_elapsed / 60:.2f} minutes"
    )

    print(
        f"Average per image  : "
        f"{avg_time:.4f} seconds"
    )

    print(
        f"Processing speed   : "
        f"{images_per_second:.2f} images/sec"
    )

    print(
        f"Processed images   : "
        f"{split_out_dir}"
    )

    print(
        f"Updated CSV        : "
        f"{csv_path}"
    )

    print(
        f"{'-' * 60}"
    )

    return {
        "split": split_name,
        "total": total_images,
        "successful": successful,
        "failed": failed,
        "elapsed": split_elapsed
    }


# =====================================================================
# BEFORE / AFTER SAMPLE
# =====================================================================

def save_before_after_sample(
    processed_root: str,
    n_samples: int = 3
):
    """
    Saves side-by-side before/after examples.

    Left:
        Original image

    Right:
        Preprocessed image

    These samples can later be used in your presentation/review.
    """

    train_csv = os.path.join(
        config.OUTPUT_DIR,
        "train_split_raw_paths_backup.csv"
    )

    if not os.path.exists(train_csv):

        print(
            "\n[WARNING] Raw training backup not found."
        )

        return

    raw_df = pd.read_csv(
        train_csv
    )

    # Make sure n_samples does not exceed dataset size.
    n_samples = min(
        n_samples,
        len(raw_df)
    )

    # Random but reproducible samples.
    raw_df = raw_df.sample(
        n=n_samples,
        random_state=config.RANDOM_SEED
    )

    sample_dir = os.path.join(
        config.OUTPUT_DIR,
        "plots",
        "preprocessing_samples"
    )

    os.makedirs(
        sample_dir,
        exist_ok=True
    )

    saved_samples = 0

    for i, row in enumerate(
        raw_df.itertuples()
    ):

        # ---------------------------------------------------------
        # Load original image
        # ---------------------------------------------------------

        original = cv2.imread(
            row.filepath
        )

        if original is None:

            print(
                f"[WARNING] Could not read "
                f"sample {row.image_id}"
            )

            continue

        # ---------------------------------------------------------
        # Resize original only for comparison.
        # ---------------------------------------------------------

        original_resized = cv2.resize(
            original,
            (
                config.IMAGE_SIZE,
                config.IMAGE_SIZE
            ),
            interpolation=cv2.INTER_AREA
        )

        # ---------------------------------------------------------
        # Process original image.
        # ---------------------------------------------------------

        processed = preprocess_image(
            row.filepath
        )

        # ---------------------------------------------------------
        # Side-by-side comparison.
        # ---------------------------------------------------------

        side_by_side = np.hstack(
            [
                original_resized,
                processed
            ]
        )

        out_path = os.path.join(
            sample_dir,
            f"before_after_{i}_{row.image_id}.jpg"
        )

        write_success = cv2.imwrite(
            out_path,
            side_by_side
        )

        if write_success:
            saved_samples += 1

    print(
        f"\nSaved {saved_samples} "
        f"before/after comparison images."
    )

    print(
        f"Location:"
        f"\n  {sample_dir}"
    )

    print(
        "\nComparison format:"
        "\n  LEFT  = original"
        "\n  RIGHT = preprocessed"
    )


# =====================================================================
# MAIN
# =====================================================================

def main():

    # ================================================================
    # TOTAL TIMER
    # ================================================================

    total_start_time = time.perf_counter()

    print("=" * 60)

    print(
        "PHASE 1 - STEP 1b: Image Preprocessing"
    )

    print(
        "(Hair Removal -> Border Crop -> CLAHE -> Resize)"
    )

    print("=" * 60)

    print(
        "\nImage size:"
        f" {config.IMAGE_SIZE} x {config.IMAGE_SIZE}"
    )

    print(
        "\nProcessing splits:"
        "\n  1. Train"
        "\n  2. Validation"
        "\n  3. Test"
    )

    print(
        "\nStarting preprocessing..."
    )

    # ================================================================
    # OUTPUT DIRECTORY
    # ================================================================

    processed_root = os.path.join(
        config.OUTPUT_DIR,
        "processed_images"
    )

    os.makedirs(
        processed_root,
        exist_ok=True
    )

    # ================================================================
    # PROCESS ALL SPLITS
    # ================================================================

    results = []

    for split_name in [
        "train",
        "val",
        "test"
    ]:

        result = process_split(
            split_name,
            processed_root
        )

        results.append(
            result
        )

    # ================================================================
    # BEFORE / AFTER SAMPLES
    # ================================================================

    save_before_after_sample(
        processed_root,
        n_samples=3
    )

    # ================================================================
    # TOTAL TIME
    # ================================================================

    total_elapsed = (
        time.perf_counter()
        - total_start_time
    )

    total_images = sum(
        result["total"]
        for result in results
    )

    total_successful = sum(
        result["successful"]
        for result in results
    )

    total_failed = sum(
        result["failed"]
        for result in results
    )

    # ================================================================
    # FINAL SUMMARY
    # ================================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "PREPROCESSING SUMMARY"
    )

    print(
        "=" * 60
    )

    print(
        f"Total images       : {total_images}"
    )

    print(
        f"Successful         : {total_successful}"
    )

    print(
        f"Failed             : {total_failed}"
    )

    print(
        f"\nTotal time         : "
        f"{total_elapsed:.2f} seconds"
    )

    print(
        f"Total time         : "
        f"{total_elapsed / 60:.2f} minutes"
    )

    if total_images > 0:

        print(
            f"Average per image  : "
            f"{total_elapsed / total_images:.4f} seconds"
        )

        print(
            f"Overall speed      : "
            f"{total_images / total_elapsed:.2f} images/sec"
        )

    print(
        f"\nProcessed images:"
        f"\n  {processed_root}"
    )

    print(
        f"\nBefore/after samples:"
        f"\n  {os.path.join(config.OUTPUT_DIR, 'plots', 'preprocessing_samples')}"
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "PHASE 1 - STEP 1b COMPLETE"
    )

    print(
        "=" * 60
    )

    print(
        "\nAll split CSVs now point to the preprocessed images."
    )

    print(
        "\nNext:"
    )

    print(
        "Run 02_dirichlet_partition.py"
    )

    print(
        "It will automatically use the preprocessed "
        "training images from train_split.csv."
    )


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    main()