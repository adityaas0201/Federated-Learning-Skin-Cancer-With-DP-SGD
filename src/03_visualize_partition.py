"""
03_visualize_partition.py
--------------------------
PHASE 1 - STEP 3

WHAT THIS DOES:
  Reads outputs/partition_summary.csv (produced by step 2) and generates
  two plots that PROVE your Dirichlet partitioning created realistic,
  skewed hospital data:

    1. A heatmap: clients (rows) x classes (columns), colored by sample
       count -- visually shows some clients have much more of certain
       classes than others.
    2. A stacked bar chart: one bar per client, showing the class
       composition of that client's data as proportions.

WHY THIS MATTERS FOR YOUR REVIEW:
  This is the single plot most likely to be asked about first: "show me
  that your hospital data is actually realistic/non-IID." These two charts
  are your direct visual evidence -- save them and put them in your
  review deck.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "configs"))
import config


def plot_heatmap(summary_df: pd.DataFrame, class_list: list, save_path: str):
    class_counts = summary_df[class_list]

    plt.figure(figsize=(10, 6))
    sns.heatmap(
        class_counts,
        annot=True,
        fmt="d",
        cmap="YlGnBu",
        linewidths=0.5,
        cbar_kws={"label": "Number of samples"},
    )
    plt.title(
        f"Per-Client Class Distribution (Dirichlet alpha={config.DIRICHLET_ALPHA})\n"
        f"Non-IID Hospital Data Simulation - HAM10000"
    )
    plt.xlabel("Skin Lesion Class")
    plt.ylabel("Simulated Hospital Client ID")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved heatmap to: {save_path}")


def plot_stacked_bar(summary_df: pd.DataFrame, class_list: list, save_path: str):
    class_counts = summary_df[class_list]
    # convert to proportions (each row sums to 1) so bars are comparable
    proportions = class_counts.div(class_counts.sum(axis=1), axis=0)

    proportions.plot(
        kind="bar",
        stacked=True,
        figsize=(10, 6),
        colormap="tab20",
    )
    plt.title(
        f"Class Composition per Simulated Hospital Client\n"
        f"(Dirichlet alpha={config.DIRICHLET_ALPHA})"
    )
    plt.xlabel("Hospital Client ID")
    plt.ylabel("Proportion of Local Data")
    plt.legend(title="Class", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved stacked bar chart to: {save_path}")


def main():
    print("=" * 60)
    print("PHASE 1 - STEP 3: Visualize Non-IID Partition")
    print("=" * 60)

    summary_path = os.path.join(config.OUTPUT_DIR, "partition_summary.csv")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(
            f"Could not find {summary_path}. Run 02_dirichlet_partition.py first."
        )

    summary_df = pd.read_csv(summary_path, index_col="client_id")

    plots_dir = os.path.join(config.OUTPUT_DIR, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    plot_heatmap(
        summary_df,
        config.CLASS_LIST,
        os.path.join(plots_dir, "partition_heatmap.png"),
    )
    plot_stacked_bar(
        summary_df,
        config.CLASS_LIST,
        os.path.join(plots_dir, "partition_stacked_bar.png"),
    )

    print("\nPHASE 1 - STEP 3 COMPLETE.")
    print(f"Open the plots in: {plots_dir}")
    print("\n>>> PHASE 1 (20%) IS NOW COMPLETE. <<<")
    print("You should now have:")
    print("  1. train/val/test CSVs split at the lesion level (no data leakage)")
    print("  2. Per-client CSVs simulating non-IID hospital data")
    print("  3. A heatmap + stacked bar chart proving realistic class skew")
    print("\nThese three outputs are your Phase 1 deliverable for review.")
    print("Next: Phase 2 (40%) - local hybrid model + FedAvg training loop.")


if __name__ == "__main__":
    main()
