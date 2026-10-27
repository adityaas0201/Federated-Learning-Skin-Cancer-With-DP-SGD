
"""
Evaluate the trained ConvNeXt-Tiny + MaxViT-Tiny hybrid ensemble
on the external ISIC 2019 dataset.

Expected project layout:
    fl_project/
        configs/config.py
        src/06_dataset.py
        src/07G_ensemble_model.py
        src/07A_model_convnext.py
        src/07F_model_maxvit_tiny.py
        src/05_handcrafted_features.py
        src/evaluate_ensemble_isic2019.py

This script evaluates an existing checkpoint. It does not train the model.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# 1. PROJECT PATHS
# ---------------------------------------------------------

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent

CONFIG_DIR = PROJECT_ROOT / "configs"
sys.path.insert(0, str(CONFIG_DIR))

import config


# External ISIC 2019 data paths from the existing project setup.
IMAGE_DIR = Path(
    r"C:\Users\arula\OneDrive\Desktop\FL Skin Cancer"
    r"\ISIC_2019_Test_Input_Preprocessed"
)

GROUND_TRUTH_CSV = Path(
    r"C:\Users\arula\OneDrive\Desktop\FL Skin Cancer"
    r"\ISIC_2019_Test_GroundTruth.csv"
)

# Best checkpoint saved by 08G_train_centralized_ensemble.py.
CHECKPOINT_PATH = Path(
    r"C:\Users\arula\OneDrive\Desktop\FL Skin Cancer\fl_project\outputs\models\ensemble_centralized\centralized_convnext_maxvit_ensemble_best.pt"
)

# Must be the scaler fitted on the training data.
SCALER_PATH = (
    Path(config.HANDCRAFTED_FEATURE_DIR)
    / "feature_scaler.pkl"
)

OUTPUT_DIR = (
    Path(config.OUTPUT_DIR)
    / "isic2019_evaluation"
    / "convnext_maxvit_ensemble"
)

CLASS_NAMES = ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"]

# The ensemble runs TWO backbones. Begin with batch size 1
# to reduce memory pressure on a 4-GB GPU.
BATCH_SIZE = 1
NUM_WORKERS = 0


# ---------------------------------------------------------
# 2. DYNAMIC MODULE LOADING
# ---------------------------------------------------------

def load_module(filename: str, module_name: str):
    """Load a Python module from a project source file."""
    source_path = HERE / filename

    if not source_path.is_file():
        raise FileNotFoundError(
            f"Required source file not found:\n{source_path}"
        )

    spec = importlib.util.spec_from_file_location(
        module_name, source_path
    )

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module: {source_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module


feature_module = load_module(
    "05_handcrafted_features.py",
    "ensemble_eval_handcrafted_features",
)

ensemble_module = load_module(
    "07G_ensemble_model.py",
    "ensemble_eval_model",
)

extract_all_handcrafted_features = (
    feature_module.extract_all_handcrafted_features
)

EnsembleClassifier = ensemble_module.ConvNeXtMaxViTEnsemble


# ---------------------------------------------------------
# 3. IMAGE PREPROCESSING
# ---------------------------------------------------------

image_size = int(config.IMAGE_SIZE)

transform = transforms.Compose(
    [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config.NORMALIZE_MEAN,
            std=config.NORMALIZE_STD,
        ),
    ]
)


# ---------------------------------------------------------
# 4. GROUND-TRUTH PREPARATION
# ---------------------------------------------------------

# ISIC 2019 ground-truth class columns and their project labels.
ISIC_TO_PROJECT = {
    "NV": "nv",
    "MEL": "mel",
    "BKL": "bkl",
    "BCC": "bcc",
    "AK": "akiec",
    "VASC": "vasc",
    "DF": "df",
    "SCC": "SCC",
    "UNK": "UNK",
}

ONE_HOT_COLUMNS = [
    "MEL", "NV", "BCC", "AK", "BKL",
    "DF", "VASC", "SCC", "UNK",
]

IMAGE_ID_CANDIDATES = [
    "image",
    "image_id",
    "image_name",
    "filename",
    "isic_id",
    "id",
]

LABEL_CANDIDATES = [
    "label",
    "dx",
    "diagnosis",
    "class",
    "target",
]


def find_column(df: pd.DataFrame, candidates: list[str]):
    """Find a column without depending on capitalization."""
    lookup = {str(col).strip().lower(): col for col in df.columns}

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    return None


def normalize_label(value):
    """Map a categorical label to one of the seven project classes."""
    if pd.isna(value):
        return None

    label = str(value).strip()

    if label in CLASS_NAMES:
        return label

    mapped = ISIC_TO_PROJECT.get(label.upper())

    return mapped


def load_ground_truth(csv_path: Path) -> pd.DataFrame:
    """
    Read categorical or one-hot ISIC ground truth.

    SCC and UNK are excluded because the project model has seven classes.
    """
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Ground-truth CSV not found:\n{csv_path}"
        )

    df = pd.read_csv(csv_path)
    df.columns = [str(col).strip() for col in df.columns]

    image_col = find_column(df, IMAGE_ID_CANDIDATES)

    if image_col is None:
        # Common ISIC format: image identifier is the first column.
        # Only use this fallback if a recognizable label representation
        # is present elsewhere in the CSV.
        image_col = df.columns[0]

    normalized_columns = {
        str(col).strip().upper(): col for col in df.columns
    }

    one_hot_available = all(
        col in normalized_columns for col in ONE_HOT_COLUMNS
    )

    rows = []

    if one_hot_available:
        for _, row in df.iterrows():
            values = {}

            for class_column in ONE_HOT_COLUMNS:
                value = pd.to_numeric(
                    row[normalized_columns[class_column]],
                    errors="coerce",
                )

                values[class_column] = (
                    float(value) if pd.notna(value) else 0.0
                )

            active = [
                class_column
                for class_column, value in values.items()
                if value > 0.5
            ]

            if len(active) != 1:
                # Do not silently assign an ambiguous target.
                continue

            raw_label = active[0]
            label = normalize_label(raw_label)

            # Exclude SCC, UNK, and anything outside the seven classes.
            if label not in CLASS_NAMES:
                continue

            rows.append(
                {
                    "image_id": str(row[image_col]).strip(),
                    "label": label,
                }
            )

    else:
        label_col = find_column(df, LABEL_CANDIDATES)

        if label_col is None:
            raise ValueError(
                "Could not identify ground-truth labels. Expected either "
                "ISIC one-hot columns (MEL, NV, BCC, AK, BKL, DF, VASC, "
                "SCC, UNK) or a categorical label/dx/diagnosis column. "
                f"CSV columns: {list(df.columns)}"
            )

        for _, row in df.iterrows():
            label = normalize_label(row[label_col])

            if label not in CLASS_NAMES:
                continue

            rows.append(
                {
                    "image_id": str(row[image_col]).strip(),
                    "label": label,
                }
            )

    result = pd.DataFrame(rows)

    if result.empty:
        raise ValueError(
            "No valid samples from the seven target classes were found "
            "in the ground-truth CSV."
        )

    # Reject duplicates rather than evaluating an ambiguous CSV.
    if result["image_id"].duplicated().any():
        duplicates = result.loc[
            result["image_id"].duplicated(keep=False), "image_id"
        ].head(10).tolist()

        raise ValueError(
            f"Duplicate image identifiers found in ground truth: "
            f"{duplicates}"
        )

    result["target"] = result["label"].map(
        {name: index for index, name in enumerate(CLASS_NAMES)}
    )

    print("\nGround-truth samples by class:")
    print(
        result["label"]
        .value_counts()
        .reindex(CLASS_NAMES, fill_value=0)
        .to_string()
    )
    print(f"Seven-class evaluation samples: {len(result)}")

    return result


# ---------------------------------------------------------
# 5. IMAGE FILE LOOKUP
# ---------------------------------------------------------

SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}


def build_image_lookup(image_dir: Path) -> dict[str, Path]:
    """Index images by their filename stem."""
    if not image_dir.is_dir():
        raise FileNotFoundError(
            f"Preprocessed image directory not found:\n{image_dir}"
        )

    lookup = {}

    for path in image_dir.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        stem = path.stem

        if stem in lookup:
            raise ValueError(
                "Multiple files have the same image identifier "
                f"{stem!r}:\n{lookup[stem]}\n{path}"
            )

        lookup[stem] = path

    if not lookup:
        raise FileNotFoundError(
            f"No supported image files found under:\n{image_dir}"
        )

    return lookup


def resolve_image_path(image_id: str, lookup: dict[str, Path]):
    """
    Resolve an image identifier.

    Handles identifiers that already include an image extension.
    """
    identifier = str(image_id).strip()
    stem = Path(identifier).stem

    return lookup.get(stem)


# ---------------------------------------------------------
# 6. EXTERNAL EVALUATION DATASET
# ---------------------------------------------------------

class ISICEnsembleDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_lookup: dict[str, Path],
        scaler,
    ):
        self.df = dataframe.reset_index(drop=True)
        self.image_lookup = image_lookup
        self.scaler = scaler

        missing = []
        self.image_paths = []

        for image_id in self.df["image_id"]:
            path = resolve_image_path(image_id, image_lookup)

            if path is None:
                missing.append(image_id)

            self.image_paths.append(path)

        if missing:
            missing_path = OUTPUT_DIR / "missing_images.csv"
            pd.DataFrame(
                {"image_id": missing}
            ).to_csv(missing_path, index=False)

            raise FileNotFoundError(
                f"{len(missing)} labeled images could not be located. "
                "Evaluation stopped to avoid silently dropping samples.\n"
                f"Missing-image list: {missing_path}\n"
                f"First missing identifiers: {missing[:10]}"
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image_path = self.image_paths[index]

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image_tensor = transform(image)

        # This must use the same feature extraction implementation
        # and the scaler fitted on training data.
        features = extract_all_handcrafted_features(
            str(image_path)
        )

        features = np.asarray(
            features, dtype=np.float32
        ).reshape(1, -1)

        expected_dim = int(config.HANDCRAFTED_FEATURE_DIM)

        if features.shape[1] != expected_dim:
            raise ValueError(
                f"Feature dimension mismatch for {image_path.name}: "
                f"expected {expected_dim}, got {features.shape[1]}"
            )

        features = self.scaler.transform(features)
        features = np.asarray(features, dtype=np.float32).reshape(-1)

        label = int(row["target"])

        return (
            image_tensor,
            torch.from_numpy(features),
            torch.tensor(label, dtype=torch.long),
            str(row["image_id"]),
        )


# ---------------------------------------------------------
# 7. CHECKPOINT LOADING
# ---------------------------------------------------------

def load_checkpoint(model, checkpoint_path: Path, device):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Ensemble checkpoint not found:\n{checkpoint_path}\n"
            "Run 08G_train_centralized_ensemble.py first if training "
            "has not completed."
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise TypeError(
            "Checkpoint does not contain a valid model state dictionary."
        )

    # Accommodate checkpoints saved from DataParallel.
    if state_dict and all(
        key.startswith("module.") for key in state_dict
    ):
        state_dict = {
            key[len("module."):]: value
            for key, value in state_dict.items()
        }

    model.load_state_dict(state_dict, strict=True)

    if isinstance(checkpoint, dict):
        print("\nCheckpoint metadata:")
        for key in (
            "epoch",
            "best_val_accuracy",
            "best_val_macro_f1",
            "class_list",
            "handcrafted_feature_dim",
            "fusion",
        ):
            if key in checkpoint:
                print(f"  {key}: {checkpoint[key]}")

    return model


# ---------------------------------------------------------
# 8. EVALUATION
# ---------------------------------------------------------

@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()

    all_labels = []
    all_predictions = []
    all_probabilities = []
    all_image_ids = []

    use_amp = device.type == "cuda"

    for images, features, labels, image_ids in tqdm(
        loader, desc="Evaluating ensemble"
    ):
        images = images.to(device, non_blocking=True)
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # The ensemble returns log probabilities. Applying softmax
        # recovers the fused probabilities for reporting.
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            log_probabilities = model(images, features)

        probabilities = torch.softmax(
            log_probabilities.float(), dim=1
        )

        predictions = probabilities.argmax(dim=1)

        all_labels.extend(labels.cpu().tolist())
        all_predictions.extend(predictions.cpu().tolist())
        all_probabilities.extend(probabilities.cpu().tolist())
        all_image_ids.extend(list(image_ids))

    return (
        np.asarray(all_labels, dtype=np.int64),
        np.asarray(all_predictions, dtype=np.int64),
        np.asarray(all_probabilities, dtype=np.float32),
        all_image_ids,
    )


# ---------------------------------------------------------
# 9. SAVE EVALUATION ARTIFACTS
# ---------------------------------------------------------

def save_confusion_matrix(y_true, y_pred, output_path: Path):
    labels = list(range(len(CLASS_NAMES)))

    cm = confusion_matrix(
        y_true, y_pred, labels=labels
    )

    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(cm)

    ax.set_xticks(labels)
    ax.set_yticks(labels)
    ax.set_xticklabels(CLASS_NAMES, rotation=40, ha="right")
    ax.set_yticklabels(CLASS_NAMES)

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("ConvNeXt-Tiny + MaxViT-Tiny Ensemble\nISIC 2019")

    threshold = cm.max() / 2 if cm.size else 0

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
                fontsize=9,
            )

    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_results(
    y_true,
    y_pred,
    probabilities,
    image_ids,
):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_f1 = f1_score(
        y_true, y_pred, average="weighted", zero_division=0
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )

    report_path = OUTPUT_DIR / "classification_report.txt"
    report_path.write_text(
        "ConvNeXt-Tiny + MaxViT-Tiny Ensemble\n"
        "External evaluation: ISIC 2019 seven-class subset\n"
        f"Number of evaluated images: {len(y_true)}\n"
        f"Accuracy: {accuracy:.6f}\n"
        f"Macro-F1: {macro_f1:.6f}\n"
        f"Weighted-F1: {weighted_f1:.6f}\n\n"
        + report,
        encoding="utf-8",
    )

    per_class_f1 = f1_score(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
        average=None,
        zero_division=0,
    )

    pd.DataFrame(
        {
            "class": CLASS_NAMES,
            "f1_score": per_class_f1,
        }
    ).to_csv(
        OUTPUT_DIR / "per_class_f1.csv",
        index=False,
    )

    predictions_df = pd.DataFrame(
        {
            "image_id": image_ids,
            "true_label": [
                CLASS_NAMES[index] for index in y_true
            ],
            "predicted_label": [
                CLASS_NAMES[index] for index in y_pred
            ],
            "correct": y_true == y_pred,
        }
    )

    for index, class_name in enumerate(CLASS_NAMES):
        predictions_df[f"prob_{class_name}"] = probabilities[:, index]

    predictions_df.to_csv(
        OUTPUT_DIR / "predictions.csv",
        index=False,
    )

    save_confusion_matrix(
        y_true,
        y_pred,
        OUTPUT_DIR / "confusion_matrix.png",
    )

    print("\n" + "=" * 65)
    print("ENSEMBLE — ISIC 2019 EXTERNAL EVALUATION")
    print("=" * 65)
    print(f"Evaluated images: {len(y_true)}")
    print(f"Accuracy:         {accuracy:.4f}")
    print(f"Macro-F1:         {macro_f1:.4f}")
    print(f"Weighted-F1:      {weighted_f1:.4f}")
    print("\nClassification report:\n")
    print(report)

    print("\nSaved artifacts:")
    print(f"  Report:          {report_path}")
    print(f"  Per-class F1:    {OUTPUT_DIR / 'per_class_f1.csv'}")
    print(f"  Predictions:     {OUTPUT_DIR / 'predictions.csv'}")
    print(f"  Confusion matrix:{OUTPUT_DIR / 'confusion_matrix.png'}")

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


# ---------------------------------------------------------
# 10. MAIN
# ---------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 65)
    print("CONVNEXT-TINY + MAXVIT-TINY ENSEMBLE EVALUATION")
    print("=" * 65)
    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"VRAM: "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GiB"
        )

    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Images:     {IMAGE_DIR}")
    print(f"Ground truth: {GROUND_TRUTH_CSV}")
    print(f"Scaler:     {SCALER_PATH}")
    print(f"Output:     {OUTPUT_DIR}")

    if not SCALER_PATH.is_file():
        raise FileNotFoundError(
            f"Training-fitted feature scaler not found:\n{SCALER_PATH}"
        )

    scaler = joblib.load(SCALER_PATH)

    ground_truth = load_ground_truth(GROUND_TRUTH_CSV)
    image_lookup = build_image_lookup(IMAGE_DIR)

    print(f"\nIndexed image files: {len(image_lookup)}")

    dataset = ISICEnsembleDataset(
        dataframe=ground_truth,
        image_lookup=image_lookup,
        scaler=scaler,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    print(f"Evaluation samples: {len(dataset)}")

    model = EnsembleClassifier(
        num_classes=int(config.NUM_CLASSES),
        handcrafted_feature_dim=int(
            config.HANDCRAFTED_FEATURE_DIM
        ),
        pretrained=False,
    )

    model = load_checkpoint(
        model, CHECKPOINT_PATH, device
    )

    model = model.to(device)
    model.eval()

    print("\nStarting external evaluation...")

    y_true, y_pred, probabilities, image_ids = evaluate(
        model, loader, device
    )

    save_results(
        y_true,
        y_pred,
        probabilities,
        image_ids,
    )

    if device.type == "cuda":
        print(
            "\nPeak GPU allocated: "
            f"{torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB"
        )

    print("\nEvaluation completed.")


if __name__ == "__main__":
    main()
