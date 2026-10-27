"""
01_prepare_data.py
------------------
PHASE 1 - STEP 1

WHAT THIS DOES:
  1. Loads HAM10000 metadata.
  2. Maps each image to its full file path across the two image folders.
  3. Splits the dataset into train / val / test at the LESION level.
  4. Verifies that no lesion appears in more than one split.
  5. Saves the resulting splits as CSV files for later scripts.

WHY THIS MATTERS FOR YOUR REVIEW:
  HAM10000 sometimes contains multiple images of the same lesion.
  If we split at the image level, images from the same lesion could
  appear in both training and testing data, causing data leakage.

  Therefore, we split using 'lesion_id' instead of 'image_id'.

  The assertions below explicitly verify that:
      Train ∩ Validation = empty
      Train ∩ Test       = empty
      Validation ∩ Test  = empty

  This provides a reproducible and auditable check that there is
  no lesion-level data leakage.
"""

import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------
# IMPORT CONFIG
# ---------------------------------------------------------------------
# Project structure:
#
# fl_project/
# ├── configs/
# │   └── config.py
# ├── scripts/
# │   └── 01_prepare_data.py
# └── outputs/
#
# Therefore, we add ../configs to Python's import path.

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "configs"
    )
)

import config


# =====================================================================
# BUILD IMAGE PATH LOOKUP
# =====================================================================

def build_image_path_lookup():
    """
    Builds a dictionary:

        image_id -> full image file path

    HAM10000 images are stored across two folders, so both folders
    are scanned.

    Example:

        ISIC_0027419
        ->
        C:/.../HAM10000_images_part_1/ISIC_0027419.jpg
    """

    lookup = {}

    for folder in config.IMAGE_DIRS:

        if not os.path.isdir(folder):
            print(
                f"[WARNING] Image folder not found: {folder}"
            )
            continue

        for fname in os.listdir(folder):

            if fname.lower().endswith(
                (".jpg", ".jpeg", ".png")
            ):

                # Remove file extension.
                image_id = os.path.splitext(fname)[0]

                lookup[image_id] = os.path.join(
                    folder,
                    fname
                )

    return lookup


# =====================================================================
# STRATIFY SAFETY CHECK
# =====================================================================

def check_stratify_feasibility(unique_lesions: pd.DataFrame, label_col: str = "label"):
    """
    sklearn's train_test_split(..., stratify=...) requires EVERY class
    to have at least 2 members in the data being split -- if a class has
    only 1 unique lesion, it cannot be divided across two output sets,
    and train_test_split will raise a ValueError.

    This is unlikely with HAM10000's main 7 classes (which each have
    many lesions), but rare classes or a filtered/subsetted version of
    the metadata could hit this edge case. We check for it up front and
    print a clear, specific warning BEFORE the split runs, rather than
    letting the script fail on a cryptic sklearn traceback that doesn't
    say which class caused the problem.
    """

    class_counts = unique_lesions[label_col].value_counts()
    too_rare = class_counts[class_counts < 2]

    if len(too_rare) > 0:
        print(
            f"\n[WARNING] The following classes have fewer than 2 unique "
            f"lesions and WILL cause stratified train_test_split to fail:"
        )
        print(too_rare.to_dict())
        print(
            "Consider merging these classes, dropping them, or switching "
            "to a non-stratified split before proceeding.\n"
        )
    else:
        print(
            "Stratify feasibility check: PASSED - every class has at "
            "least 2 unique lesions. ✓"
        )


# =====================================================================
# MAIN
# =====================================================================

def main():

    print("=" * 60)
    print("PHASE 1 - STEP 1: Data Preparation")
    print("=" * 60)

    # -----------------------------------------------------------------
    # 1. LOAD METADATA
    # -----------------------------------------------------------------

    if not os.path.exists(config.METADATA_FILE):

        raise FileNotFoundError(
            f"Could not find metadata file at "
            f"{config.METADATA_FILE}\n"
            f"Check RAW_DATA_DIR and METADATA_FILE "
            f"in configs/config.py"
        )

    # HAM10000_metadata.tab from Harvard Dataverse is tab-separated.
    # Some mirrors provide a .csv (comma-separated) version instead.
    # We detect the separator automatically based on file extension,
    # so either format works without manual renaming or conversion.

    file_ext = os.path.splitext(
        config.METADATA_FILE
    )[1].lower()

    separator = (
        "\t"
        if file_ext == ".tab"
        else ","
    )

    separator_name = (
        "tab"
        if separator == "\t"
        else "comma"
    )

    meta = pd.read_csv(
        config.METADATA_FILE,
        sep=separator
    )

    print(
        f"Loaded metadata "
        f"({separator_name}-separated): "
        f"{len(meta)} rows"
    )

    print(
        f"Columns: {list(meta.columns)}"
    )

    # -----------------------------------------------------------------
    # 2. BASIC METADATA VALIDATION
    # -----------------------------------------------------------------

    required_columns = {
        "image_id",
        "lesion_id",
        "dx"
    }

    missing_columns = (
        required_columns
        - set(meta.columns)
    )

    if missing_columns:

        raise ValueError(
            f"Missing required metadata columns: "
            f"{missing_columns}"
        )

    # Every image should have a unique image_id.

    if meta["image_id"].nunique() != len(meta):

        raise ValueError(
            "Duplicate image_id values found in metadata."
        )

    # -----------------------------------------------------------------
    # 3. ATTACH FULL IMAGE FILE PATHS
    # -----------------------------------------------------------------

    path_lookup = build_image_path_lookup()

    print(
        f"Found {len(path_lookup)} image files on disk"
    )

    meta["filepath"] = meta["image_id"].map(
        path_lookup
    )

    missing = meta["filepath"].isna().sum()

    if missing > 0:

        print(
            f"[WARNING] {missing} rows have no "
            f"matching image file - dropping them"
        )

        meta = meta.dropna(
            subset=["filepath"]
        ).reset_index(drop=True)

    else:

        print(
            "All metadata images matched "
            "to image files. ✓"
        )

    # -----------------------------------------------------------------
    # 4. KEEP ONLY THE COLUMNS WE NEED
    # -----------------------------------------------------------------

    meta = meta[
        [
            "image_id",
            "lesion_id",
            "dx",
            "filepath"
        ]
    ].rename(
        columns={
            "dx": "label"
        }
    )

    # -----------------------------------------------------------------
    # 5. CHECK LABELS
    # -----------------------------------------------------------------

    unknown_labels = (
        set(meta["label"].unique())
        - set(config.CLASS_LIST)
    )

    if unknown_labels:

        print(
            f"[WARNING] Unknown labels found "
            f"and will be dropped: {unknown_labels}"
        )

        meta = meta[
            meta["label"].isin(
                config.CLASS_LIST
            )
        ].reset_index(drop=True)

    print(
        "\nClass distribution "
        "(full dataset):"
    )

    print(
        meta["label"].value_counts()
    )

    # -----------------------------------------------------------------
    # 6. CREATE UNIQUE LESION DATASET
    # -----------------------------------------------------------------
    #
    # Important:
    #
    # HAM10000 can contain multiple images belonging to the same lesion.
    #
    # Therefore we first create a list of unique lesions.
    #
    # The lesion label is used only for stratification.

    unique_lesions = (
        meta[
            ["lesion_id"]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    lesion_label = (
        meta
        .drop_duplicates("lesion_id")
        [["lesion_id", "label"]]
    )

    unique_lesions = unique_lesions.merge(
        lesion_label,
        on="lesion_id"
    )

    print(
        f"\nUnique lesions: "
        f"{len(unique_lesions)}"
    )

    # -----------------------------------------------------------------
    # 6b. STRATIFY FEASIBILITY CHECK (guard before splitting)
    # -----------------------------------------------------------------
    #
    # train_test_split(..., stratify=...) requires every class to have
    # at least 2 members, otherwise it cannot place that class into
    # both output sets and will raise a ValueError. We check this here
    # so that, if it were ever to happen, you get a clear, specific
    # warning naming the problem class BEFORE the split runs -- not a
    # confusing sklearn traceback partway through the script.

    check_stratify_feasibility(unique_lesions, label_col="label")

    # -----------------------------------------------------------------
    # 7. LESION-LEVEL TRAIN / TEST SPLIT
    # -----------------------------------------------------------------

    train_val_lesions, test_lesions = train_test_split(
        unique_lesions,
        test_size=config.TEST_SPLIT,
        stratify=unique_lesions["label"],
        random_state=config.RANDOM_SEED
    )

    # -----------------------------------------------------------------
    # 8. LESION-LEVEL TRAIN / VALIDATION SPLIT
    # -----------------------------------------------------------------

    train_lesions, val_lesions = train_test_split(
        train_val_lesions,
        test_size=config.VAL_SPLIT,
        stratify=train_val_lesions["label"],
        random_state=config.RANDOM_SEED
    )

    # -----------------------------------------------------------------
    # 9. CREATE SETS OF LESION IDs
    # -----------------------------------------------------------------

    train_ids = set(
        train_lesions["lesion_id"]
    )

    val_ids = set(
        val_lesions["lesion_id"]
    )

    test_ids = set(
        test_lesions["lesion_id"]
    )

    # -----------------------------------------------------------------
    # 10. VERIFY NO LESION-LEVEL DATA LEAKAGE
    # -----------------------------------------------------------------

    # A lesion must belong to exactly one split.
    #
    # isdisjoint() returns True when the two sets have NO common
    # elements.
    #
    # assert stops the program if the condition is False.

    assert train_ids.isdisjoint(
        val_ids
    ), "DATA LEAKAGE: Train and validation share lesions!"

    assert train_ids.isdisjoint(
        test_ids
    ), "DATA LEAKAGE: Train and test share lesions!"

    assert val_ids.isdisjoint(
        test_ids
    ), "DATA LEAKAGE: Validation and test share lesions!"

    print(
        "\nLesion-level leakage check: PASSED ✓"
    )

    print(
        "  Train ∩ Validation = empty ✓"
    )

    print(
        "  Train ∩ Test       = empty ✓"
    )

    print(
        "  Validation ∩ Test  = empty ✓"
    )

    # -----------------------------------------------------------------
    # 11. CREATE TRAIN / VALIDATION / TEST DATAFRAMES
    # -----------------------------------------------------------------

    train_df = (
        meta[
            meta["lesion_id"].isin(
                train_ids
            )
        ]
        .reset_index(drop=True)
    )

    val_df = (
        meta[
            meta["lesion_id"].isin(
                val_ids
            )
        ]
        .reset_index(drop=True)
    )

    test_df = (
        meta[
            meta["lesion_id"].isin(
                test_ids
            )
        ]
        .reset_index(drop=True)
    )

    # -----------------------------------------------------------------
    # 12. DISPLAY SPLIT SIZES
    # -----------------------------------------------------------------

    print(
        "\nSplit sizes "
        "(by IMAGE count, after lesion-level split):"
    )

    print(
        f"  Train: {len(train_df)}"
    )

    print(
        f"  Val:   {len(val_df)}"
    )

    print(
        f"  Test:  {len(test_df)}"
    )

    print(
        f"  Total: "
        f"{len(train_df) + len(val_df) + len(test_df)}"
    )

    # -----------------------------------------------------------------
    # 13. DISPLAY CLASS DISTRIBUTION
    # -----------------------------------------------------------------

    print(
        "\nTrain class distribution:"
    )

    print(
        train_df["label"].value_counts()
    )

    print(
        "\nValidation class distribution:"
    )

    print(
        val_df["label"].value_counts()
    )

    print(
        "\nTest class distribution:"
    )

    print(
        test_df["label"].value_counts()
    )

    # -----------------------------------------------------------------
    # 14. SAVE SPLITS
    # -----------------------------------------------------------------

    train_path = os.path.join(
        config.OUTPUT_DIR,
        "train_split.csv"
    )

    val_path = os.path.join(
        config.OUTPUT_DIR,
        "val_split.csv"
    )

    test_path = os.path.join(
        config.OUTPUT_DIR,
        "test_split.csv"
    )

    train_df.to_csv(
        train_path,
        index=False
    )

    val_df.to_csv(
        val_path,
        index=False
    )

    test_df.to_csv(
        test_path,
        index=False
    )

    print(
        f"\nSaved train split:"
        f"\n  {train_path}"
    )

    print(
        f"Saved validation split:"
        f"\n  {val_path}"
    )

    print(
        f"Saved test split:"
        f"\n  {test_path}"
    )

    # -----------------------------------------------------------------
    # 15. FINAL VALIDATION
    # -----------------------------------------------------------------

    # Verify that every image is assigned to exactly one split.

    total_split_images = (
        len(train_df)
        + len(val_df)
        + len(test_df)
    )

    assert total_split_images == len(meta), (
        "Split image counts do not add up "
        "to the original dataset size!"
    )

    # Verify image IDs don't overlap.

    assert set(
        train_df["image_id"]
    ).isdisjoint(
        set(val_df["image_id"])
    )

    assert set(
        train_df["image_id"]
    ).isdisjoint(
        set(test_df["image_id"])
    )

    assert set(
        val_df["image_id"]
    ).isdisjoint(
        set(test_df["image_id"])
    )

    print(
        "\nFinal validation: PASSED ✓"
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "PHASE 1 - STEP 1 COMPLETE"
    )

    print(
        "=" * 60
    )

    print(
        "\nNext:"
    )

    print(
        "Run 02_dirichlet_partition.py to split "
        "the TRAIN data across simulated hospitals."
    )


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    main()