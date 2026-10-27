"""
05_handcrafted_features.py
----------------------------
PHASE 2 - STEP 5

Handcrafted feature extraction for the hybrid skin-lesion classifier.

RUNS AFTER:
    01_prepare_data.py
    01b_preprocess_images.py

WHAT THIS DOES:
    Extracts four types of handcrafted features from every preprocessed
    HAM10000 image:

        1. GLCM  -> texture
        2. LBP   -> local texture
        3. HOG   -> edges / shape / border
        4. Color + Asymmetry -> ABCD-inspired features

    The features are computed ONCE and cached to disk as .npz files.

WHY CACHE?
    These are CPU-based classical computer-vision operations. Recomputing
    them inside every training epoch would unnecessarily slow training.

OUTPUT:
    outputs/
        handcrafted_features/
            train_features.npz
            val_features.npz
            test_features.npz

IMPORTANT:
    Feature standardization is NOT performed here.

    Later, during the training pipeline:
        - fit scaler using TRAIN features only
        - apply the same scaler to validation/test features

    This prevents data leakage.
"""

import os
import sys
import time

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from skimage.feature import (
    graycomatrix,
    graycoprops,
    local_binary_pattern,
    hog,
)


# =====================================================================
# LOAD CONFIG
# =====================================================================

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "configs"
    )
)

import config


# =====================================================================
# GLCM SETTINGS
# =====================================================================

# We quantize the grayscale image from 256 levels to 32 levels before
# calculating GLCM.
#
# Why?
# 256 levels creates a 256 x 256 co-occurrence matrix.
# 32 levels creates a 32 x 32 matrix.
#
# This is substantially cheaper while still providing useful texture
# information for this application.

GLCM_LEVELS = 32


# =====================================================================
# FEATURE NAME HELPERS
# =====================================================================

def get_feature_names():
    """
    Creates names for every element of the final handcrafted vector.

    The order MUST exactly match the order used when concatenating
    the actual features.
    """

    names = []

    # ---------------------------------------------------------------
    # GLCM
    # ---------------------------------------------------------------

    for prop in config.GLCM_PROPERTIES:
        names.append(f"GLCM_{prop}")

    # ---------------------------------------------------------------
    # LBP
    # ---------------------------------------------------------------

    num_lbp_bins = config.LBP_POINTS + 2

    for i in range(num_lbp_bins):
        names.append(f"LBP_{i}")

    # ---------------------------------------------------------------
    # HOG
    #
    # HOG dimension depends on the configuration, so these names are
    # generated later from the actual HOG output.
    # ---------------------------------------------------------------

    # Placeholder; replaced by the actual HOG dimension in
    # build_feature_names().
    
    # ---------------------------------------------------------------
    # Color
    # ---------------------------------------------------------------

    names.extend([
        "HSV_H_mean",
        "HSV_H_std",
        "HSV_S_mean",
        "HSV_S_std",
        "HSV_V_mean",
        "HSV_V_std",
    ])

    # ---------------------------------------------------------------
    # Asymmetry
    # ---------------------------------------------------------------

    names.extend([
        "Asymmetry_left_right",
        "Asymmetry_top_bottom",
    ])

    return names


# =====================================================================
# STEP A — GLCM
# =====================================================================

def quantize_grayscale(
    gray_image: np.ndarray,
    levels: int = GLCM_LEVELS
) -> np.ndarray:
    """
    Converts an 8-bit grayscale image from 256 intensity levels
    into a smaller number of intensity levels.

    Example:

        0-255
          ↓
        0-31

    This makes GLCM considerably more memory/compute efficient.
    """

    if gray_image.dtype != np.uint8:
        gray_image = np.clip(
            gray_image,
            0,
            255
        ).astype(np.uint8)

    quantized = (
        gray_image.astype(np.uint16) * levels // 256
    ).astype(np.uint8)

    # Safety guarantee:
    # values must be in [0, levels-1]
    quantized = np.clip(
        quantized,
        0,
        levels - 1
    )

    return quantized


def extract_glcm_features(
    gray_image: np.ndarray
) -> np.ndarray:
    """
    Extracts GLCM texture features.

    Uses the distances and angles specified in config.py.

    For each configured property:
        - calculate value for every angle
        - average across distances and angles

    Returns:
        1D float32 array.

    With the current config:
        6 GLCM features.
    """

    gray_quantized = quantize_grayscale(
        gray_image,
        GLCM_LEVELS
    )

    glcm = graycomatrix(
        gray_quantized,
        distances=config.GLCM_DISTANCES,
        angles=config.GLCM_ANGLES,
        levels=GLCM_LEVELS,
        symmetric=True,
        normed=True,
    )

    features = []

    for prop in config.GLCM_PROPERTIES:

        values = graycoprops(
            glcm,
            prop
        )

        # Average across distance and angle dimensions
        value = values.mean()

        features.append(value)

    features = np.asarray(
        features,
        dtype=np.float32
    )

    return features


# =====================================================================
# STEP B — LBP
# =====================================================================

def extract_lbp_features(
    gray_image: np.ndarray
) -> np.ndarray:
    """
    Extracts Local Binary Pattern features.

    Uses:
        LBP_RADIUS
        LBP_POINTS
        LBP_METHOD

    For uniform LBP:

        number of bins = P + 2

    With the current configuration:

        P = 8
        bins = 10

    The histogram is normalized so that its values sum approximately
    to 1.
    """

    lbp = local_binary_pattern(
        gray_image,
        P=config.LBP_POINTS,
        R=config.LBP_RADIUS,
        method=config.LBP_METHOD,
    )

    num_bins = config.LBP_POINTS + 2

    histogram, _ = np.histogram(
        lbp.ravel(),
        bins=num_bins,
        range=(0, num_bins),
    )

    histogram = histogram.astype(
        np.float32
    )

    histogram /= (
        histogram.sum() + 1e-8
    )

    return histogram


# =====================================================================
# STEP C — HOG
# =====================================================================

def extract_hog_features(
    gray_image: np.ndarray
) -> np.ndarray:
    """
    Extracts Histogram of Oriented Gradients.

    HOG captures:
        - edge orientations
        - local gradients
        - structural information
        - lesion border characteristics

    The feature dimension is determined automatically by the
    HOG configuration.
    """

    features = hog(
        gray_image,
        orientations=config.HOG_ORIENTATIONS,
        pixels_per_cell=config.HOG_PIXELS_PER_CELL,
        cells_per_block=config.HOG_CELLS_PER_BLOCK,
        block_norm="L2-Hys",
        feature_vector=True,
    )

    return features.astype(
        np.float32
    )


# =====================================================================
# STEP D — COLOR + ASYMMETRY
# =====================================================================

def extract_color_asymmetry_features(
    image_bgr: np.ndarray
) -> np.ndarray:
    """
    Extracts:

        1. HSV color statistics
           - mean and standard deviation
           - H, S and V channels

        2. Color asymmetry
           - left vs right
           - top vs bottom

    These provide computational proxies for the Color and Asymmetry
    components of the ABCD rule.

    Returns:
        8-dimensional vector.
    """

    hsv = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2HSV
    )

    h, w = hsv.shape[:2]

    # ---------------------------------------------------------------
    # GROUP 1 — GLOBAL HSV STATISTICS
    # ---------------------------------------------------------------

    color_stats = []

    for channel in range(3):

        channel_data = hsv[:, :, channel].astype(
            np.float32
        )

        color_stats.append(
            channel_data.mean()
        )

        color_stats.append(
            channel_data.std()
        )

    color_stats = np.asarray(
        color_stats,
        dtype=np.float32
    )

    # ---------------------------------------------------------------
    # GROUP 2 — IMAGE HALVES
    # ---------------------------------------------------------------

    left_half = hsv[
        :,
        :w // 2,
        :
    ]

    right_half = hsv[
        :,
        w // 2:,
        :
    ]

    top_half = hsv[
        :h // 2,
        :,
        :
    ]

    bottom_half = hsv[
        h // 2:,
        :,
        :
    ]

    # ---------------------------------------------------------------
    # HUE HISTOGRAM
    # ---------------------------------------------------------------

    def half_hist(
        half_img,
        bins=16
    ):

        hist, _ = np.histogram(
            half_img[:, :, 0],
            bins=bins,
            range=(0, 180)
        )

        hist = hist.astype(
            np.float32
        )

        hist /= (
            hist.sum() + 1e-8
        )

        return hist

    # ---------------------------------------------------------------
    # HISTOGRAM DIFFERENCE
    # ---------------------------------------------------------------

    def hist_diff(
        hist_a,
        hist_b
    ):

        return np.sum(
            np.abs(
                hist_a - hist_b
            )
        )

    lr_asymmetry = hist_diff(
        half_hist(left_half),
        half_hist(right_half)
    )

    tb_asymmetry = hist_diff(
        half_hist(top_half),
        half_hist(bottom_half)
    )

    asymmetry_features = np.asarray(
        [
            lr_asymmetry,
            tb_asymmetry
        ],
        dtype=np.float32
    )

    return np.concatenate(
        [
            color_stats,
            asymmetry_features
        ]
    )


# =====================================================================
# FULL FEATURE PIPELINE
# =====================================================================

def extract_all_handcrafted_features(
    filepath: str
) -> np.ndarray:
    """
    Loads one preprocessed image and generates the complete feature
    vector:

        GLCM
        +
        LBP
        +
        HOG
        +
        Color
        +
        Asymmetry
    """

    image_bgr = cv2.imread(
        filepath
    )

    if image_bgr is None:

        raise ValueError(
            f"Could not read image:\n{filepath}"
        )

    # Convert to grayscale for texture/shape features
    gray = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2GRAY
    )

    # Extract each feature group
    glcm_features = extract_glcm_features(
        gray
    )

    lbp_features = extract_lbp_features(
        gray
    )

    hog_features = extract_hog_features(
        gray
    )

    color_features = extract_color_asymmetry_features(
        image_bgr
    )

    # ---------------------------------------------------------------
    # COMBINE
    # ---------------------------------------------------------------

    combined = np.concatenate(
        [
            glcm_features,
            lbp_features,
            hog_features,
            color_features
        ]
    )

    combined = combined.astype(
        np.float32
    )

    # ---------------------------------------------------------------
    # NUMERICAL SAFETY CHECK
    # ---------------------------------------------------------------

    if not np.all(
        np.isfinite(combined)
    ):

        raise ValueError(
            f"NaN or Inf detected in handcrafted features:\n"
            f"{filepath}"
        )

    return combined


# =====================================================================
# BUILD FEATURE NAMES
# =====================================================================

def build_feature_names(
    feature_dim: int
):
    """
    Builds feature names in exactly the same order as the concatenated
    feature vector.

    HOG dimension is calculated automatically because it depends on
    the HOG configuration.
    """

    names = []

    # ---------------------------------------------------------------
    # GLCM
    # ---------------------------------------------------------------

    for prop in config.GLCM_PROPERTIES:

        names.append(
            f"GLCM_{prop}"
        )

    # ---------------------------------------------------------------
    # LBP
    # ---------------------------------------------------------------

    for i in range(
        config.LBP_POINTS + 2
    ):

        names.append(
            f"LBP_{i}"
        )

    # ---------------------------------------------------------------
    # Determine HOG dimension
    # ---------------------------------------------------------------

    hog_dim = (
        feature_dim
        - len(config.GLCM_PROPERTIES)
        - (config.LBP_POINTS + 2)
        - 8
    )

    for i in range(hog_dim):

        names.append(
            f"HOG_{i}"
        )

    # ---------------------------------------------------------------
    # Color
    # ---------------------------------------------------------------

    names.extend(
        [
            "HSV_H_mean",
            "HSV_H_std",
            "HSV_S_mean",
            "HSV_S_std",
            "HSV_V_mean",
            "HSV_V_std",
        ]
    )

    # ---------------------------------------------------------------
    # Asymmetry
    # ---------------------------------------------------------------

    names.extend(
        [
            "Asymmetry_left_right",
            "Asymmetry_top_bottom",
        ]
    )

    # Final verification
    if len(names) != feature_dim:

        raise RuntimeError(
            f"Feature-name mismatch: "
            f"{len(names)} names vs "
            f"{feature_dim} features"
        )

    return names


# =====================================================================
# PROCESS ONE SPLIT
# =====================================================================

def process_split(
    split_name: str
):
    """
    Processes one split:

        train
        val
        test

    Reads the split CSV, extracts features for every image, validates
    the result, and saves one compressed .npz file.
    """

    csv_path = os.path.join(
        config.OUTPUT_DIR,
        f"{split_name}_split.csv"
    )

    if not os.path.exists(csv_path):

        raise FileNotFoundError(
            f"\nCould not find:\n"
            f"{csv_path}\n\n"
            f"Run 01_prepare_data.py and "
            f"01b_preprocess_images.py first."
        )

    df = pd.read_csv(
        csv_path
    )

    print("\n" + "=" * 60)
    print(
        f"PROCESSING {split_name.upper()} SPLIT"
    )
    print("=" * 60)

    print(
        f"Images: {len(df)}"
    )

    # ---------------------------------------------------------------
    # CHECK IMAGE IDS
    # ---------------------------------------------------------------

    if df["image_id"].duplicated().any():

        duplicates = (
            df.loc[
                df["image_id"].duplicated(),
                "image_id"
            ]
            .tolist()
        )

        raise ValueError(
            f"Duplicate image_id values found in "
            f"{split_name} split.\n"
            f"Examples: {duplicates[:10]}"
        )

    image_ids = []
    feature_vectors = []

    failed_images = []

    start_time = time.perf_counter()

    # ---------------------------------------------------------------
    # EXTRACT FEATURES
    # ---------------------------------------------------------------

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc=split_name
    ):

        image_id = row["image_id"]
        filepath = row["filepath"]

        try:

            features = extract_all_handcrafted_features(
                filepath
            )

            image_ids.append(
                image_id
            )

            feature_vectors.append(
                features
            )

        except Exception as e:

            failed_images.append(
                {
                    "image_id": image_id,
                    "filepath": filepath,
                    "error": str(e)
                }
            )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    # ---------------------------------------------------------------
    # STRICT FAILURE CHECK
    # ---------------------------------------------------------------

    if failed_images:

        print(
            f"\nERROR: "
            f"{len(failed_images)} image(s) failed."
        )

        for item in failed_images[:10]:

            print(
                f"  {item['image_id']}: "
                f"{item['error']}"
            )

        raise RuntimeError(
            f"\nFeature extraction stopped.\n"
            f"{len(failed_images)} image(s) failed in "
            f"the {split_name} split.\n"
            f"Fix the problem and rerun the script."
        )

    # ---------------------------------------------------------------
    # SAFETY CHECK
    # ---------------------------------------------------------------

    if len(feature_vectors) != len(df):

        raise RuntimeError(
            f"Feature count mismatch for {split_name}:\n"
            f"CSV images     : {len(df)}\n"
            f"Feature vectors: {len(feature_vectors)}"
        )

    # ---------------------------------------------------------------
    # STACK FEATURES
    # ---------------------------------------------------------------

    feature_matrix = np.stack(
        feature_vectors,
        axis=0
    ).astype(
        np.float32
    )

    image_ids_array = np.asarray(
        image_ids
    )

    # ---------------------------------------------------------------
    # CHECK SHAPE
    # ---------------------------------------------------------------

    if feature_matrix.shape[0] != len(df):

        raise RuntimeError(
            f"Row mismatch after stacking:\n"
            f"Expected: {len(df)}\n"
            f"Got: {feature_matrix.shape[0]}"
        )

    # ---------------------------------------------------------------
    # CHECK NUMERICAL VALUES
    # ---------------------------------------------------------------

    if not np.all(
        np.isfinite(feature_matrix)
    ):

        raise RuntimeError(
            f"NaN/Inf values found in "
            f"{split_name} feature matrix."
        )

    # ---------------------------------------------------------------
    # FEATURE DIMENSION
    # ---------------------------------------------------------------

    feature_dim = feature_matrix.shape[1]

    feature_names = build_feature_names(
        feature_dim
    )

    # ---------------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------------

    os.makedirs(
        config.HANDCRAFTED_FEATURE_DIR,
        exist_ok=True
    )

    out_path = os.path.join(
        config.HANDCRAFTED_FEATURE_DIR,
        f"{split_name}_features.npz"
    )

    np.savez_compressed(
        out_path,
        image_ids=image_ids_array,
        features=feature_matrix,
        feature_names=np.asarray(
            feature_names
        )
    )

    # ---------------------------------------------------------------
    # PERFORMANCE
    # ---------------------------------------------------------------

    images_per_second = (
        len(df) / elapsed
        if elapsed > 0
        else 0
    )

    # ---------------------------------------------------------------
    # REPORT
    # ---------------------------------------------------------------

    print("\n" + "-" * 60)
    print(
        f"{split_name.upper()} FEATURE EXTRACTION COMPLETE"
    )
    print("-" * 60)

    print(
        f"Total images       : {len(df)}"
    )

    print(
        f"Successful         : {len(image_ids)}"
    )

    print(
        f"Failed             : {len(failed_images)}"
    )

    print(
        f"Feature dimension  : {feature_dim}"
    )

    print(
        f"Time taken         : "
        f"{elapsed:.2f} seconds "
        f"({elapsed / 60:.2f} min)"
    )

    print(
        f"Processing speed   : "
        f"{images_per_second:.2f} images/sec"
    )

    print(
        f"Saved to           : {out_path}"
    )

    print("-" * 60)

    return feature_dim


# =====================================================================
# VERIFY SAVED FEATURE FILE
# =====================================================================

def verify_feature_file(
    split_name: str
):
    """
    Reopens the saved .npz file and verifies that the cache is readable
    and internally consistent.
    """

    path = os.path.join(
        config.HANDCRAFTED_FEATURE_DIR,
        f"{split_name}_features.npz"
    )

    data = np.load(
        path,
        allow_pickle=False
    )

    image_ids = data["image_ids"]
    features = data["features"]
    feature_names = data["feature_names"]

    if len(image_ids) != features.shape[0]:

        raise RuntimeError(
            f"{split_name}: image ID count does not match "
            f"feature row count."
        )

    if features.shape[1] != len(feature_names):

        raise RuntimeError(
            f"{split_name}: feature name count does not "
            f"match feature dimension."
        )

    if len(set(image_ids.tolist())) != len(image_ids):

        raise RuntimeError(
            f"{split_name}: duplicate image IDs found "
            f"in saved feature cache."
        )

    if not np.all(
        np.isfinite(features)
    ):

        raise RuntimeError(
            f"{split_name}: NaN/Inf found in saved cache."
        )

    print(
        f"Verified {split_name}: "
        f"{features.shape[0]} images × "
        f"{features.shape[1]} features ✓"
    )


# =====================================================================
# MAIN
# =====================================================================

def main():

    total_start = time.perf_counter()

    print("=" * 60)
    print(
        "PHASE 2 - STEP 5: "
        "HANDCRAFTED FEATURE EXTRACTION"
    )
    print("=" * 60)

    print(
        "\nFeature groups:"
    )

    print(
        f"  GLCM             : "
        f"{len(config.GLCM_PROPERTIES)} features"
    )

    print(
        f"  LBP              : "
        f"{config.LBP_POINTS + 2} features"
    )

    print(
        "  HOG              : calculated automatically"
    )

    print(
        "  Color/Asymmetry  : 8 features"
    )

    print(
        f"\nGLCM gray levels   : "
        f"{GLCM_LEVELS}"
    )

    print(
        f"Output directory   : "
        f"{config.HANDCRAFTED_FEATURE_DIR}"
    )

    print(
        "\nImportant:"
    )

    print(
        "  Features are cached to disk."
    )

    print(
        "  No feature standardization is performed here."
    )

    print(
        "  Scaling will be fitted on TRAIN only later."
    )

    # ---------------------------------------------------------------
    # PROCESS ALL SPLITS
    # ---------------------------------------------------------------

    feature_dims = []

    for split_name in [
        "train",
        "val",
        "test"
    ]:

        dim = process_split(
            split_name
        )

        feature_dims.append(
            dim
        )

    # ---------------------------------------------------------------
    # ALL SPLITS MUST HAVE SAME DIMENSION
    # ---------------------------------------------------------------

    if len(set(feature_dims)) != 1:

        raise RuntimeError(
            f"\nFeature dimension mismatch!\n"
            f"Train/Val/Test dimensions: "
            f"{feature_dims}"
        )

    feature_dim = feature_dims[0]

    # ---------------------------------------------------------------
    # VERIFY SAVED FILES
    # ---------------------------------------------------------------

    print(
        "\nVerifying saved feature caches..."
    )

    for split_name in [
        "train",
        "val",
        "test"
    ]:

        verify_feature_file(
            split_name
        )

    # ---------------------------------------------------------------
    # BREAKDOWN
    # ---------------------------------------------------------------

    glcm_dim = len(
        config.GLCM_PROPERTIES
    )

    lbp_dim = (
        config.LBP_POINTS + 2
    )

    color_asymmetry_dim = 8

    hog_dim = (
        feature_dim
        - glcm_dim
        - lbp_dim
        - color_asymmetry_dim
    )

    # ---------------------------------------------------------------
    # TOTAL TIME
    # ---------------------------------------------------------------

    total_elapsed = (
        time.perf_counter()
        - total_start
    )

    # ---------------------------------------------------------------
    # FINAL REPORT
    # ---------------------------------------------------------------

    print("\n" + "=" * 60)
    print(
        "PHASE 2 - STEP 5 COMPLETE"
    )
    print("=" * 60)

    print(
        f"\nFinal handcrafted feature dimension: "
        f"{feature_dim}"
    )

    print(
        "\nFeature breakdown:"
    )

    print(
        f"  GLCM             : {glcm_dim}"
    )

    print(
        f"  LBP              : {lbp_dim}"
    )

    print(
        f"  HOG              : {hog_dim}"
    )

    print(
        f"  Color/Asymmetry  : "
        f"{color_asymmetry_dim}"
    )

    print(
        f"\nTotal images processed: "
        f"{8112 + 879 + 1024}"
    )

    print(
        f"Total time: "
        f"{total_elapsed:.2f} seconds "
        f"({total_elapsed / 60:.2f} minutes)"
    )

    print(
        "\nGenerated files:"
    )

    print(
        f"  {config.HANDCRAFTED_FEATURE_DIR}"
    )

    print(
        "    ├── train_features.npz"
    )

    print(
        "    ├── val_features.npz"
    )

    print(
        "    └── test_features.npz"
    )

    print(
        "\nAll feature caches verified successfully ✓"
    )

    print(
        "\nNext:"
    )

    print(
        "  06_dataset.py"
    )

    print(
        "  This will connect:"
    )

    print(
        "    image → image tensor"
    )

    print(
        "    image_id → handcrafted feature vector"
    )

    print(
        "    label → class index"
    )

    print(
        "\n" + "=" * 60
    )


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    main()