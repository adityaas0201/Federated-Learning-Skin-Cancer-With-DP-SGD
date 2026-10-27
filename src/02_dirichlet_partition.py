"""
02_dirichlet_partition.py
--------------------------
PHASE 1 - STEP 2

Lesion-level Dirichlet Non-IID partitioning.

Only TRAIN data is partitioned.
Validation and TEST remain centralized.

Important:
All images belonging to the same lesion_id stay in the same client.
This prevents lesion overlap between simulated hospitals.
"""

import os
import sys
import time

import numpy as np
import pandas as pd

# Import config.py
sys.path.append(
    os.path.join(os.path.dirname(__file__), "..", "configs")
)
import config


# ================================================================
# LESION-LEVEL DIRICHLET PARTITION
# ================================================================

def dirichlet_partition(df, num_clients, alpha, seed):
    """
    Partition TRAIN data at lesion level using Dirichlet distribution.

    For each class:
        1. Get unique lesions.
        2. Shuffle lesions.
        3. Sample Dirichlet proportions.
        4. Assign whole lesions to clients.
        5. All images of those lesions follow the client.
    """

    if num_clients < 2:
        raise ValueError("num_clients must be >= 2")

    if alpha <= 0:
        raise ValueError("alpha must be > 0")

    required = {"image_id", "lesion_id", "label"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    rng = np.random.default_rng(seed)

    # Store lesion IDs assigned to each client
    client_lesions = {
        i: [] for i in range(num_clients)
    }

    # ------------------------------------------------------------
    # Process every class independently
    # ------------------------------------------------------------

    for label in config.CLASS_LIST:

        class_df = df[df["label"] == label]

        if class_df.empty:
            continue

        # IMPORTANT: unique lesions, NOT individual images
        lesions = (
            class_df["lesion_id"]
            .drop_duplicates()
            .to_numpy()
        )

        rng.shuffle(lesions)

        n_lesions = len(lesions)

        # Dirichlet class proportions
        proportions = rng.dirichlet(
            np.full(num_clients, alpha)
        )

        # Convert proportions to integer lesion counts
        counts = np.floor(
            proportions * n_lesions
        ).astype(int)

        # Distribute remaining lesions
        remaining = n_lesions - counts.sum()

        if remaining > 0:
            fractional = (
                proportions * n_lesions - counts
            )

            order = np.argsort(-fractional)

            for i in range(remaining):
                counts[order[i]] += 1

        assert counts.sum() == n_lesions

        # Assign complete lesions
        start = 0

        for client_id, count in enumerate(counts):

            end = start + count

            client_lesions[client_id].extend(
                lesions[start:end].tolist()
            )

            start = end

    # ------------------------------------------------------------
    # Convert lesion assignments back to image dataframes
    # ------------------------------------------------------------

    client_dfs = {}

    for client_id in range(num_clients):

        client_df = df[
            df["lesion_id"].isin(
                client_lesions[client_id]
            )
        ].copy()

        # Shuffle images inside each client
        client_df = client_df.sample(
            frac=1,
            random_state=seed + client_id
        ).reset_index(drop=True)

        client_dfs[client_id] = client_df

    return client_dfs


# ================================================================
# SUMMARY
# ================================================================

def create_summary(client_dfs):

    rows = []

    for client_id, df in client_dfs.items():

        counts = (
            df["label"]
            .value_counts()
            .reindex(
                config.CLASS_LIST,
                fill_value=0
            )
        )

        row = {
            "client_id": client_id,
            "total_samples": len(df),
            "unique_lesions": df["lesion_id"].nunique()
        }

        row.update(counts.to_dict())
        rows.append(row)

    return pd.DataFrame(rows).set_index("client_id")


def create_percentage_summary(client_dfs):

    rows = []

    for client_id, df in client_dfs.items():

        counts = (
            df["label"]
            .value_counts(normalize=True)
            .reindex(
                config.CLASS_LIST,
                fill_value=0
            )
            * 100
        )

        row = {
            "client_id": client_id,
            "total_samples": len(df)
        }

        row.update(counts.round(2).to_dict())
        rows.append(row)

    return pd.DataFrame(rows).set_index("client_id")


# ================================================================
# VALIDATION
# ================================================================

def validate_partition(original_df, client_dfs):

    print("\n" + "=" * 60)
    print("PARTITION VALIDATION")
    print("=" * 60)

    # ------------------------------------------------------------
    # 1. No empty clients
    # ------------------------------------------------------------

    empty = [
        cid for cid, df in client_dfs.items()
        if len(df) == 0
    ]

    if empty:
        raise ValueError(
            f"Empty clients detected: {empty}"
        )

    print("Empty-client check: PASSED ✓")

    # ------------------------------------------------------------
    # 2. Sample conservation
    # ------------------------------------------------------------

    total = sum(
        len(df) for df in client_dfs.values()
    )

    if total != len(original_df):
        raise ValueError(
            f"Sample mismatch: "
            f"original={len(original_df)}, "
            f"clients={total}"
        )

    print(f"Total sample count: {total}")
    print("Sample conservation check: PASSED ✓")

    # ------------------------------------------------------------
    # 3. Image overlap
    # ------------------------------------------------------------

    image_ids = []

    for df in client_dfs.values():
        image_ids.extend(df["image_id"].tolist())

    if len(image_ids) != len(set(image_ids)):
        raise ValueError(
            "DATA LEAKAGE: image appears in multiple clients!"
        )

    print("Image-level overlap check: PASSED ✓")

    # ------------------------------------------------------------
    # 4. Lesion overlap
    # ------------------------------------------------------------

    lesion_owner = {}

    for client_id, df in client_dfs.items():

        for lesion_id in df["lesion_id"].unique():

            if lesion_id in lesion_owner:

                raise ValueError(
                    "LESION-LEVEL OVERLAP DETECTED!\n"
                    f"Lesion {lesion_id} appears in "
                    f"Client {lesion_owner[lesion_id]} "
                    f"and Client {client_id}."
                )

            lesion_owner[lesion_id] = client_id

    print("Lesion-level overlap check: PASSED ✓")

    # ------------------------------------------------------------
    # 5. Every original image assigned
    # ------------------------------------------------------------

    original_images = set(
        original_df["image_id"]
    )

    assigned_images = set(image_ids)

    if original_images != assigned_images:
        raise ValueError(
            "Some original images were not assigned "
            "to any client!"
        )

    print("Complete image assignment check: PASSED ✓")

    print("\nAll partition validation checks PASSED ✓")


# ================================================================
# SAVE CONFIG
# ================================================================

def save_config(path, train_size):

    with open(path, "w", encoding="utf-8") as f:

        f.write("FEDERATED LEARNING PARTITION CONFIGURATION\n")
        f.write("=" * 48 + "\n\n")
        f.write("Dataset: HAM10000\n")
        f.write(f"Training samples: {train_size}\n")
        f.write(
            f"Number of clients: {config.NUM_CLIENTS}\n"
        )
        f.write(
            f"Dirichlet alpha: {config.DIRICHLET_ALPHA}\n"
        )
        f.write(
            f"Random seed: {config.RANDOM_SEED}\n"
        )
        f.write(
            "Partition level: LESION\n"
        )
        f.write(
            "Validation: centralized\n"
        )
        f.write(
            "Test: centralized\n"
        )


# ================================================================
# MAIN
# ================================================================

def main():

    start_time = time.perf_counter()

    print("=" * 60)
    print("PHASE 1 - STEP 2: Dirichlet Non-IID Partitioning")
    print("=" * 60)

    print("\nConfiguration:")
    print(f"  Clients : {config.NUM_CLIENTS}")
    print(f"  Alpha   : {config.DIRICHLET_ALPHA}")
    print(f"  Seed    : {config.RANDOM_SEED}")
    print("  Level   : LESION")

    print("\nOnly TRAIN is partitioned.")
    print("Validation and TEST remain untouched.")

    # ------------------------------------------------------------
    # Load training CSV
    # ------------------------------------------------------------

    train_csv = os.path.join(
        config.OUTPUT_DIR,
        "train_split.csv"
    )

    if not os.path.exists(train_csv):
        raise FileNotFoundError(
            f"{train_csv} not found.\n"
            "Run 01_prepare_data.py first."
        )

    train_df = pd.read_csv(train_csv)

    print(
        f"\nLoaded training data: "
        f"{len(train_df)} images"
    )

    print(
        f"Unique training lesions: "
        f"{train_df['lesion_id'].nunique()}"
    )

    # ------------------------------------------------------------
    # Validate input
    # ------------------------------------------------------------

    required = {
        "image_id",
        "lesion_id",
        "label",
        "filepath"
    }

    missing = required - set(train_df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    if train_df["image_id"].nunique() != len(train_df):
        raise ValueError(
            "Duplicate image_id values found."
        )

    unknown = (
        set(train_df["label"].unique())
        - set(config.CLASS_LIST)
    )

    if unknown:
        raise ValueError(
            f"Unknown labels found: {unknown}"
        )

    print("\nOriginal training class distribution:")
    print(train_df["label"].value_counts())

    # ------------------------------------------------------------
    # Partition
    # ------------------------------------------------------------

    print("\nGenerating lesion-level Dirichlet partition...")

    client_dfs = dirichlet_partition(
        train_df,
        num_clients=config.NUM_CLIENTS,
        alpha=config.DIRICHLET_ALPHA,
        seed=config.RANDOM_SEED
    )

    print("Partition generated ✓")

    # ------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------

    validate_partition(
        train_df,
        client_dfs
    )

    # ------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------

    partition_dir = os.path.join(
        config.OUTPUT_DIR,
        "client_partitions"
    )

    os.makedirs(
        partition_dir,
        exist_ok=True
    )

    # ------------------------------------------------------------
    # Save client CSVs
    # ------------------------------------------------------------

    print("\nSaving client partitions:")

    for client_id, df in client_dfs.items():

        path = os.path.join(
            partition_dir,
            f"client_{client_id}.csv"
        )

        df.to_csv(path, index=False)

        print(
            f"  Client {client_id}: "
            f"{len(df)} images, "
            f"{df['lesion_id'].nunique()} lesions"
        )

    # ------------------------------------------------------------
    # Save summaries
    # ------------------------------------------------------------

    summary = create_summary(client_dfs)

    summary_path = os.path.join(
        config.OUTPUT_DIR,
        "partition_summary.csv"
    )

    summary.to_csv(summary_path)

    percentages = create_percentage_summary(
        client_dfs
    )

    percentage_path = os.path.join(
        config.OUTPUT_DIR,
        "partition_percentages.csv"
    )

    percentages.to_csv(
        percentage_path
    )

    # ------------------------------------------------------------
    # Display results
    # ------------------------------------------------------------

    print("\n" + "=" * 60)
    print("PER-CLIENT CLASS DISTRIBUTION (COUNTS)")
    print("=" * 60)

    print(summary.to_string())

    print("\n" + "=" * 60)
    print("PER-CLIENT CLASS DISTRIBUTION (%)")
    print("=" * 60)

    print(
        percentages.to_string()
    )

    # ------------------------------------------------------------
    # Save configuration
    # ------------------------------------------------------------

    config_path = os.path.join(
        config.OUTPUT_DIR,
        "partition_config.txt"
    )

    save_config(
        config_path,
        len(train_df)
    )

    # ------------------------------------------------------------
    # Timing
    # ------------------------------------------------------------

    elapsed = time.perf_counter() - start_time

    print("\n" + "=" * 60)
    print("PHASE 1 - STEP 2 COMPLETE")
    print("=" * 60)

    print(f"\nTraining samples : {len(train_df)}")
    print(f"Clients          : {config.NUM_CLIENTS}")
    print(f"Alpha            : {config.DIRICHLET_ALPHA}")
    print(f"Partition level  : LESION")
    print(f"Time taken       : {elapsed:.4f} seconds")

    print(f"\nClient CSVs:")
    print(f"  {partition_dir}")

    print(f"\nSummary:")
    print(f"  {summary_path}")

    print(f"\nPercentages:")
    print(f"  {percentage_path}")

    print(f"\nConfig:")
    print(f"  {config_path}")

    print("\nNext:")
    print("Run:")
    print("  python src\\03_visualize_partition.py")


if __name__ == "__main__":
    main()