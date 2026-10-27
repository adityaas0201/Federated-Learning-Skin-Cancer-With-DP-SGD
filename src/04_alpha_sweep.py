"""
04_alpha_sweep.py
-----------------
PHASE 1 - OPTIONAL EXPERIMENT

Compares different Dirichlet alpha values using the SAME:
    - TRAIN split
    - random seed
    - number of clients
    - lesion-level partitioning

This allows fair comparison of client heterogeneity.

Alpha values:
    0.1  -> extreme non-IID
    0.3  -> strong non-IID
    1.0  -> moderate non-IID
    100  -> approximately IID

IMPORTANT:
Partitioning is performed at LESION level.
Therefore, images belonging to the same lesion cannot be
distributed across different clients.
"""

import os
import sys
import time
import importlib.util

import pandas as pd
import matplotlib.pyplot as plt


# ================================================================
# CONFIG
# ================================================================

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "configs"
    )
)

import config


# ================================================================
# LOAD STEP 2 FUNCTIONS
# ================================================================

def load_partition_module():

    path = os.path.join(
        os.path.dirname(__file__),
        "02_dirichlet_partition.py"
    )

    spec = importlib.util.spec_from_file_location(
        "partition_module",
        path
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


partition_module = load_partition_module()

dirichlet_partition = (
    partition_module.dirichlet_partition
)

create_summary = (
    partition_module.create_summary
)

validate_partition = (
    partition_module.validate_partition
)


# ================================================================
# ALPHA VALUES
# ================================================================

ALPHA_VALUES_TO_TEST = [
    0.1,
    0.3,
    1.0,
    100.0
]


# ================================================================
# HEATMAP
# ================================================================

def plot_heatmap(
    summary_df,
    alpha,
    save_path
):

    classes = config.CLASS_LIST

    data = summary_df[
        classes
    ]

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    image = ax.imshow(
        data.values,
        aspect="auto"
    )

    ax.set_xticks(
        range(len(classes))
    )

    ax.set_xticklabels(
        classes
    )

    ax.set_yticks(
        range(len(data.index))
    )

    ax.set_yticklabels(
        [
            f"Client {i}"
            for i in data.index
        ]
    )

    ax.set_xlabel(
        "Skin Lesion Class"
    )

    ax.set_ylabel(
        "Hospital Client"
    )

    ax.set_title(
        f"Client Class Distribution — "
        f"Dirichlet α={alpha}"
    )

    # Add numbers
    for r in range(data.shape[0]):

        for c in range(data.shape[1]):

            ax.text(
                c,
                r,
                str(data.iloc[r, c]),
                ha="center",
                va="center",
                fontsize=9
            )

    colorbar = fig.colorbar(
        image,
        ax=ax
    )

    colorbar.set_label(
        "Number of Images"
    )

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


# ================================================================
# STACKED BAR
# ================================================================

def plot_stacked_bar(
    summary_df,
    alpha,
    save_path
):

    classes = config.CLASS_LIST

    counts = summary_df[
        classes
    ]

    proportions = counts.div(
        counts.sum(axis=1),
        axis=0
    )

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    bottom = pd.Series(
        0.0,
        index=proportions.index
    )

    for label in classes:

        values = proportions[
            label
        ]

        ax.bar(
            proportions.index.astype(str),
            values,
            bottom=bottom,
            label=label
        )

        bottom += values

    ax.set_title(
        f"Class Composition per Hospital — "
        f"Dirichlet α={alpha}"
    )

    ax.set_xlabel(
        "Hospital Client ID"
    )

    ax.set_ylabel(
        "Proportion of Local Data"
    )

    ax.set_ylim(
        0,
        1
    )

    ax.set_yticks(
        [0, 0.25, 0.50, 0.75, 1.0]
    )

    ax.set_yticklabels(
        ["0%", "25%", "50%", "75%", "100%"]
    )

    ax.legend(
        title="Class",
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


# ================================================================
# SUMMARY METRICS
# ================================================================

def calculate_metrics(
    summary_df,
    alpha
):

    classes = config.CLASS_LIST

    counts = summary_df[
        classes
    ]

    client_sizes = (
        summary_df["total_samples"]
    )

    # Largest and smallest client
    largest_client = int(
        client_sizes.max()
    )

    smallest_client = int(
        client_sizes.min()
    )

    # Size ratio
    size_ratio = (
        largest_client /
        smallest_client
        if smallest_client > 0
        else float("inf")
    )

    # Largest class proportion
    proportions = counts.div(
        counts.sum(axis=1),
        axis=0
    )

    max_class_prop = (
        proportions.max().max()
    )

    min_class_prop = (
        proportions.min().min()
    )

    return {
        "alpha": alpha,
        "largest_client": largest_client,
        "smallest_client": smallest_client,
        "client_size_ratio": round(
            size_ratio,
            2
        ),
        "largest_class_proportion": round(
            max_class_prop * 100,
            2
        ),
        "smallest_class_proportion": round(
            min_class_prop * 100,
            2
        )
    }


# ================================================================
# RUN ONE ALPHA
# ================================================================

def run_for_alpha(
    train_df,
    alpha
):

    print("\n" + "-" * 60)
    print(
        f"Running lesion-level partition "
        f"for alpha = {alpha}"
    )
    print("-" * 60)

    alpha_dir = os.path.join(
        config.OUTPUT_DIR,
        "alpha_sweep",
        f"alpha_{alpha}"
    )

    partitions_dir = os.path.join(
        alpha_dir,
        "client_partitions"
    )

    plots_dir = os.path.join(
        alpha_dir,
        "plots"
    )

    os.makedirs(
        partitions_dir,
        exist_ok=True
    )

    os.makedirs(
        plots_dir,
        exist_ok=True
    )

    # ------------------------------------------------------------
    # LESION-LEVEL DIRICHLET PARTITION
    # ------------------------------------------------------------

    client_dfs = dirichlet_partition(
        train_df,
        num_clients=config.NUM_CLIENTS,
        alpha=alpha,
        seed=config.RANDOM_SEED
    )

    # ------------------------------------------------------------
    # VALIDATE
    # ------------------------------------------------------------

    validate_partition(
        train_df,
        client_dfs
    )

    # ------------------------------------------------------------
    # SAVE CLIENTS
    # ------------------------------------------------------------

    for client_id, df in client_dfs.items():

        path = os.path.join(
            partitions_dir,
            f"client_{client_id}.csv"
        )

        df.to_csv(
            path,
            index=False
        )

    # ------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------

    summary_df = create_summary(
        client_dfs
    )

    summary_df.to_csv(
        os.path.join(
            alpha_dir,
            "partition_summary.csv"
        )
    )

    # ------------------------------------------------------------
    # PLOTS
    # ------------------------------------------------------------

    plot_heatmap(
        summary_df,
        alpha,
        os.path.join(
            plots_dir,
            "partition_heatmap.png"
        )
    )

    plot_stacked_bar(
        summary_df,
        alpha,
        os.path.join(
            plots_dir,
            "partition_stacked_bar.png"
        )
    )

    # ------------------------------------------------------------
    # METRICS
    # ------------------------------------------------------------

    metrics = calculate_metrics(
        summary_df,
        alpha
    )

    print(
        f"Largest client : "
        f"{metrics['largest_client']} images"
    )

    print(
        f"Smallest client: "
        f"{metrics['smallest_client']} images"
    )

    print(
        f"Client size ratio: "
        f"{metrics['client_size_ratio']}x"
    )

    print(
        f"Largest class proportion: "
        f"{metrics['largest_class_proportion']}%"
    )

    print(
        f"Smallest class proportion: "
        f"{metrics['smallest_class_proportion']}%"
    )

    print(
        f"Saved to: {alpha_dir}"
    )

    return metrics


# ================================================================
# MAIN
# ================================================================

def main():

    start_time = time.perf_counter()

    print("=" * 60)
    print("PHASE 1 - OPTIONAL: DIRICHLET ALPHA SWEEP")
    print("=" * 60)

    print(
        "\nThis experiment compares:"
    )

    for alpha in ALPHA_VALUES_TO_TEST:
        print(f"  α = {alpha}")

    print(
        "\nAll experiments use:"
    )

    print(
        f"  Clients      : {config.NUM_CLIENTS}"
    )

    print(
        f"  Random seed  : {config.RANDOM_SEED}"
    )

    print(
        "  Partition    : LESION LEVEL"
    )

    # ------------------------------------------------------------
    # Load training data
    # ------------------------------------------------------------

    train_csv = os.path.join(
        config.OUTPUT_DIR,
        "train_split.csv"
    )

    if not os.path.exists(train_csv):

        raise FileNotFoundError(
            f"\nCould not find:\n{train_csv}\n\n"
            "Run 01_prepare_data.py first."
        )

    train_df = pd.read_csv(
        train_csv
    )

    print(
        f"\nTraining images : "
        f"{len(train_df)}"
    )

    print(
        f"Training lesions: "
        f"{train_df['lesion_id'].nunique()}"
    )

    # ------------------------------------------------------------
    # Run all alpha values
    # ------------------------------------------------------------

    all_metrics = []

    for alpha in ALPHA_VALUES_TO_TEST:

        metrics = run_for_alpha(
            train_df,
            alpha
        )

        all_metrics.append(
            metrics
        )

    # ------------------------------------------------------------
    # Save comparison table
    # ------------------------------------------------------------

    results_df = pd.DataFrame(
        all_metrics
    )

    sweep_dir = os.path.join(
        config.OUTPUT_DIR,
        "alpha_sweep"
    )

    os.makedirs(
        sweep_dir,
        exist_ok=True
    )

    results_path = os.path.join(
        sweep_dir,
        "alpha_comparison.csv"
    )

    results_df.to_csv(
        results_path,
        index=False
    )

    # ------------------------------------------------------------
    # Final output
    # ------------------------------------------------------------

    elapsed = (
        time.perf_counter()
        - start_time
    )

    print("\n" + "=" * 60)
    print("ALPHA SWEEP COMPLETE")
    print("=" * 60)

    print("\nComparison:")
    print(
        results_df.to_string(
            index=False
        )
    )

    print(
        f"\nSaved comparison table:"
        f"\n  {results_path}"
    )

    print(
        "\nGenerated folders:"
    )

    for alpha in ALPHA_VALUES_TO_TEST:

        print(
            f"  outputs/alpha_sweep/"
            f"alpha_{alpha}/"
        )

    print(
        f"\nTotal time: "
        f"{elapsed:.4f} seconds"
    )

    print("\nInterpretation:")
    print(
        "  Smaller alpha → stronger client heterogeneity."
    )

    print(
        "  Larger alpha → more balanced client distributions."
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "Do not select alpha only from the plots."
    )

    print(
        "Use the client-size and class-proportion results "
        "together with training stability in Phase 2."
    )


if __name__ == "__main__":
    main()