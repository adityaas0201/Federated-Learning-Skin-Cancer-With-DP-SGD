
from __future__ import annotations

import importlib.util
import sys
import traceback
from pathlib import Path

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_fscore_support,
)
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm.auto import tqdm


# ============================================================
# 1. PATHS AND SETTINGS
# ============================================================

PROJECT_DIR = Path(
    r"C:\Users\arula\OneDrive\Desktop\FL Skin Cancer\fl_project"
)

SRC_DIR = PROJECT_DIR / "src"
CONFIG_DIR = PROJECT_DIR / "configs"

IMAGE_DIR = Path(
    r"C:\Users\arula\OneDrive\Desktop\FL Skin Cancer"
    r"\ISIC_2019_Test_Input_Preprocessed"
)

GROUND_TRUTH_CSV = Path(
    r"C:\Users\arula\OneDrive\Desktop\FL Skin Cancer"
    r"\ISIC_2019_Test_GroundTruth.csv"
)

# The uploaded MobileNet training script saves this filename
# inside config.MODEL_SAVE_DIR.
OUTPUT_DIR = (
    PROJECT_DIR
    / "outputs"
    / "isic2019_evaluation"
    / "mobilenet_v3_small"
)

MODEL_NAME = "MobileNetV3-Small Hybrid"

BATCH_SIZE = 4
NUM_WORKERS = 0  # Windows-safe.

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

for directory in (PROJECT_DIR, SRC_DIR, CONFIG_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

# Match the import style used in your training script.
import config


# ============================================================
# 2. CHECKPOINT AND CLASS CONFIGURATION
# ============================================================

CHECKPOINT_PATH = Path(
    r"C:\Users\arula\OneDrive\Desktop\FL Skin Cancer\fl_project\outputs\models\mobilenet_centralized\centralized_baseline.pt"
)

CLASS_NAMES = [
    str(name).strip().lower()
    for name in config.CLASS_LIST
]

ELIGIBLE_CLASSES = {
    "nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"
}

if len(CLASS_NAMES) != 7 or set(CLASS_NAMES) != ELIGIBLE_CLASSES:
    raise ValueError(
        "config.CLASS_LIST must contain exactly these seven classes: "
        f"{sorted(ELIGIBLE_CLASSES)}. Found: {CLASS_NAMES}"
    )

ISIC_LABEL_MAP = {
    "MEL": "mel",
    "NV": "nv",
    "BCC": "bcc",
    "AK": "akiec",
    "BKL": "bkl",
    "DF": "df",
    "VASC": "vasc",
    "SCC": "scc",
    "UNK": "unk",
}


# ============================================================
# 3. DYNAMIC MODULE LOADER
# ============================================================

def load_module_from_file(module_name: str, file_path: Path):
    if not file_path.is_file():
        raise FileNotFoundError(
            f"Required Python module not found: {file_path}"
        )

    spec = importlib.util.spec_from_file_location(
        module_name, file_path
    )

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module: {file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module


# Reuse your project's existing handcrafted feature extractor.
feature_module = load_module_from_file(
    "mobilenet_handcrafted_features",
    SRC_DIR / "05_handcrafted_features.py",
)


# ============================================================
# 4. GROUND-TRUTH CSV
# ============================================================

def normalize_image_id(value) -> str:
    return Path(str(value).strip()).stem


def normalize_label(value) -> str:
    label = str(value).strip().upper()
    return ISIC_LABEL_MAP.get(label, label.lower())


def find_image_column(df: pd.DataFrame) -> str:
    candidates = (
        "image",
        "image_id",
        "isic_id",
        "filename",
        "file_name",
    )

    lookup = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for candidate in candidates:
        if candidate in lookup:
            return lookup[candidate]

    raise ValueError(
        "Could not identify the image ID column. "
        f"CSV columns: {list(df.columns)}"
    )


def get_ground_truth(csv_path: Path) -> pd.DataFrame:
    """
    Supports either:
      - A categorical label column.
      - ISIC 2019 one-hot diagnosis columns.

    SCC and UNK are excluded from seven-class evaluation.
    """

    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Ground-truth CSV not found: {csv_path}"
        )

    df = pd.read_csv(csv_path)

    if df.empty:
        raise ValueError("The ground-truth CSV is empty.")

    image_column = find_image_column(df)

    column_lookup = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    label_column = None

    for candidate in ("label", "diagnosis", "class", "target"):
        if candidate in column_lookup:
            label_column = column_lookup[candidate]
            break

    if label_column is not None:
        result = pd.DataFrame({
            "image_id": df[image_column].map(normalize_image_id),
            "label": df[label_column].map(normalize_label),
        })

    else:
        uppercase_columns = {
            str(column).strip().upper(): column
            for column in df.columns
        }

        present = {
            key: uppercase_columns[key]
            for key in ISIC_LABEL_MAP
            if key in uppercase_columns
        }

        if not present:
            raise ValueError(
                "No categorical label column or recognized ISIC "
                f"one-hot columns found. Columns: {list(df.columns)}"
            )

        def decode_one_hot(row):
            active_labels = []

            for source_label, project_label in ISIC_LABEL_MAP.items():
                if source_label not in present:
                    continue

                value = pd.to_numeric(
                    row[present[source_label]],
                    errors="coerce",
                )

                if pd.notna(value) and value == 1:
                    active_labels.append(project_label)

            if len(active_labels) != 1:
                return "invalid"

            return active_labels[0]

        result = pd.DataFrame({
            "image_id": df[image_column].map(normalize_image_id),
            "label": df.apply(decode_one_hot, axis=1),
        })

    if result["image_id"].duplicated().any():
        duplicates = result.loc[
            result["image_id"].duplicated(keep=False),
            "image_id",
        ].unique()

        raise ValueError(
            "Duplicate image IDs in ground truth: "
            f"{duplicates[:10].tolist()}"
        )

    result = result[
        result["label"].isin(ELIGIBLE_CLASSES)
    ].copy()

    if result.empty:
        raise ValueError(
            "No eligible seven-class labels remain. "
            "Check the CSV format and label names."
        )

    return result.reset_index(drop=True)


# ============================================================
# 5. MATCH IMAGE FILES
# ============================================================

def find_image_path(
    image_id: str,
    image_dir: Path,
) -> Path | None:

    for extension in (
        ".jpg", ".jpeg", ".png", ".bmp", ".webp"
    ):
        candidate = image_dir / f"{image_id}{extension}"

        if candidate.is_file():
            return candidate

    candidate = image_dir / image_id

    if candidate.is_file():
        return candidate

    return None


# ============================================================
# 6. HANDCRAFTED FEATURES
# ============================================================

def extract_handcrafted_features(image_path: Path) -> np.ndarray:
    if not hasattr(
        feature_module,
        "extract_all_handcrafted_features",
    ):
        raise AttributeError(
            "05_handcrafted_features.py must expose "
            "extract_all_handcrafted_features()."
        )

    features = feature_module.extract_all_handcrafted_features(
        str(image_path)
    )

    features = np.asarray(
        features,
        dtype=np.float32,
    ).reshape(-1)

    expected_dim = int(config.HANDCRAFTED_FEATURE_DIM)

    if features.shape[0] != expected_dim:
        raise ValueError(
            f"Feature dimension mismatch for {image_path.name}: "
            f"expected {expected_dim}, got {features.shape[0]}."
        )

    if not np.isfinite(features).all():
        raise ValueError(
            f"Non-finite handcrafted features: {image_path.name}"
        )

    return features


# ============================================================
# 7. ISIC EVALUATION DATASET
# ============================================================

class ISICEvaluationDataset(Dataset):

    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_dir: Path,
        scaler,
    ):
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.scaler = scaler

        self.class_to_index = {
            name: index
            for index, name in enumerate(CLASS_NAMES)
        }

        self.image_transform = transforms.Compose([
            transforms.Resize((
                int(config.IMAGE_SIZE),
                int(config.IMAGE_SIZE),
            )),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config.NORMALIZE_MEAN,
                std=config.NORMALIZE_STD,
            ),
        ])

        self.samples = []
        missing = []

        for _, row in self.dataframe.iterrows():
            image_id = str(row["image_id"])
            label = str(row["label"])

            image_path = find_image_path(
                image_id,
                self.image_dir,
            )

            if image_path is None:
                missing.append(image_id)
                continue

            self.samples.append({
                "image_id": image_id,
                "image_path": image_path,
                "label": label,
            })

        if missing:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

            missing_path = OUTPUT_DIR / "missing_images.csv"

            pd.DataFrame({
                "image_id": missing,
            }).to_csv(missing_path, index=False)

            raise FileNotFoundError(
                f"{len(missing)} labelled images were not found in "
                f"{image_dir}. Missing IDs saved to {missing_path}. "
                "Evaluation stopped to avoid silently dropping samples."
            )

        if not self.samples:
            raise ValueError("No matching evaluation images found.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        image_path = sample["image_path"]

        image_bgr = cv2.imread(str(image_path))

        if image_bgr is None:
            raise RuntimeError(
                f"OpenCV could not read image: {image_path}"
            )

        image_rgb = cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2RGB,
        )

        image = Image.fromarray(image_rgb)
        image_tensor = self.image_transform(image)

        # Extract features using the same project feature extractor.
        handcrafted = extract_handcrafted_features(image_path)

        # Apply the scaler fitted on the training data.
        # Never fit a new scaler on the external evaluation dataset.
        handcrafted = self.scaler.transform(
            handcrafted.reshape(1, -1)
        )[0]

        if not np.isfinite(handcrafted).all():
            raise ValueError(
                f"Non-finite scaled features: {image_path.name}"
            )

        handcrafted_tensor = torch.tensor(
            handcrafted,
            dtype=torch.float32,
        )

        label_index = self.class_to_index[sample["label"]]

        return (
            image_tensor,
            handcrafted_tensor,
            label_index,
            sample["image_id"],
        )


# ============================================================
# 8. LOAD TRAINING-FITTED SCALER
# ============================================================

def load_feature_scaler():
    scaler_path = (
        Path(config.HANDCRAFTED_FEATURE_DIR)
        / "feature_scaler.pkl"
    )

    if not scaler_path.is_file():
        raise FileNotFoundError(
            f"Training-fitted feature scaler not found: {scaler_path}\n"
            "Check config.HANDCRAFTED_FEATURE_DIR and ensure the scaler "
            "saved during training is available."
        )

    scaler = joblib.load(scaler_path)

    if not hasattr(scaler, "transform"):
        raise TypeError(
            f"The loaded scaler does not provide transform(): "
            f"{scaler_path}"
        )

    return scaler


# ============================================================
# 9. BUILD MOBILE NET V3-SMALL HYBRID MODEL
# ============================================================

def build_model():
    model_file = SRC_DIR / "07_model.py"

    module = load_module_from_file(
        "hybrid_mobilenet_v3_small_model",
        model_file,
    )

    if not hasattr(module, "HybridClassifier"):
        raise AttributeError(
            f"{model_file} does not expose HybridClassifier."
        )

    # Match the constructor defaults in your uploaded model file.
    # pretrained=False avoids downloading ImageNet weights.
    model = module.HybridClassifier(
        num_classes=len(CLASS_NAMES),
        handcrafted_feature_dim=int(
            config.HANDCRAFTED_FEATURE_DIM
        ),
        pretrained=False,
        head_hidden_dim=512,
        dropout=0.30,
    )

    return model


# ============================================================
# 10. LOAD TRAINED CHECKPOINT
# ============================================================

def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: Path,
):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"MobileNet checkpoint not found: {checkpoint_path}\n"
            "Your uploaded training script saves "
            "'centralized_baseline.pt' under config.MODEL_SAVE_DIR. "
            "Confirm that centralized training completed successfully."
        )

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        # Compatibility with PyTorch versions without weights_only.
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
        )

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Unexpected checkpoint format; expected a dictionary."
        )

    state_dict = checkpoint.get("model_state_dict")

    # Also support a raw state_dict if one was saved separately.
    if state_dict is None:
        if all(
            isinstance(value, torch.Tensor)
            for value in checkpoint.values()
        ):
            state_dict = checkpoint
        else:
            raise ValueError(
                "Checkpoint does not contain 'model_state_dict'. "
                f"Available keys: {list(checkpoint.keys())}"
            )

    cleaned_state_dict = {}

    for key, value in state_dict.items():
        new_key = (
            key[len("module."):]
            if key.startswith("module.")
            else key
        )
        cleaned_state_dict[new_key] = value

    # Strict loading detects model/checkpoint architecture mismatches.
    model.load_state_dict(cleaned_state_dict, strict=True)

    model = model.to(DEVICE)
    model.eval()

    print(f"\nLoaded checkpoint: {checkpoint_path}")

    if "epoch" in checkpoint:
        print(f"Checkpoint epoch: {checkpoint['epoch']}")

    if "val_macro_f1" in checkpoint:
        print(
            "Recorded best validation Macro-F1: "
            f"{checkpoint['val_macro_f1']:.4f}"
        )

    if "val_acc" in checkpoint:
        print(
            "Recorded validation accuracy: "
            f"{checkpoint['val_acc']:.4f}"
        )

    print(f"Evaluation device: {DEVICE}")

    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    return model


# ============================================================
# 11. MODEL FORWARD PASS
# ============================================================

def get_logits(
    model,
    images: torch.Tensor,
    handcrafted_features: torch.Tensor,
) -> torch.Tensor:

    # This matches the uploaded MobileNet HybridClassifier.forward().
    output = model(images, handcrafted_features)

    if isinstance(output, (tuple, list)):
        output = output[0]

    if isinstance(output, dict):
        for key in ("logits", "output", "predictions"):
            if key in output:
                output = output[key]
                break
        else:
            raise ValueError(
                "Model output dictionary has no recognized prediction key."
            )

    if not isinstance(output, torch.Tensor):
        raise TypeError(
            f"Unexpected model output type: {type(output)}"
        )

    if output.ndim != 2:
        raise ValueError(
            f"Expected [batch, classes] output, got {output.shape}"
        )

    if output.shape[1] != len(CLASS_NAMES):
        raise ValueError(
            f"Expected {len(CLASS_NAMES)} output classes, "
            f"received {output.shape[1]}."
        )

    return output


# ============================================================
# 12. RUN INFERENCE
# ============================================================

@torch.inference_mode()
def run_inference(model, dataloader):
    y_true = []
    y_pred = []
    image_ids = []

    progress = tqdm(
        dataloader,
        total=len(dataloader),
        desc="Evaluating MobileNetV3-Small",
        unit="batch",
        dynamic_ncols=True,
    )

    for images, handcrafted, labels, batch_image_ids in progress:
        images = images.to(
            DEVICE,
            non_blocking=(DEVICE.type == "cuda"),
        )

        handcrafted = handcrafted.to(
            DEVICE,
            non_blocking=(DEVICE.type == "cuda"),
        )

        outputs = get_logits(
            model,
            images,
            handcrafted,
        )

        predictions = outputs.argmax(dim=1)

        y_true.extend(labels.numpy().tolist())
        y_pred.extend(predictions.cpu().numpy().tolist())
        image_ids.extend(list(batch_image_ids))

        progress.set_postfix(
            images=f"{len(y_true)}/{len(dataloader.dataset)}"
        )

    return (
        np.asarray(y_true, dtype=np.int64),
        np.asarray(y_pred, dtype=np.int64),
        image_ids,
    )


# ============================================================
# 13. SAVE EVALUATION METRICS
# ============================================================

def save_evaluation_outputs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    labels = list(range(len(CLASS_NAMES)))

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )

    report_path = OUTPUT_DIR / "classification_report.txt"

    with open(report_path, "w", encoding="utf-8") as file:
        file.write(f"Model: {MODEL_NAME}\n")
        file.write(f"Checkpoint: {CHECKPOINT_PATH}\n")
        file.write(f"Device: {DEVICE}\n")
        file.write(f"Evaluated samples: {len(y_true)}\n")
        file.write(f"Class order: {CLASS_NAMES}\n\n")
        file.write(report)
        file.write("\n")

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels,
            zero_division=0,
        )
    )

    per_class_df = pd.DataFrame({
        "class": CLASS_NAMES,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "support": support.astype(int),
    })

    per_class_path = OUTPUT_DIR / "per_class_f1.csv"
    per_class_df.to_csv(per_class_path, index=False)

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    fig, ax = plt.subplots(figsize=(10, 8))

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CLASS_NAMES,
    )

    display.plot(
        ax=ax,
        xticks_rotation=45,
        values_format="d",
        colorbar=False,
    )

    ax.set_title("Confusion Matrix — MobileNetV3-Small")
    fig.tight_layout()

    confusion_path = OUTPUT_DIR / "confusion_matrix.png"

    fig.savefig(
        confusion_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    accuracy = accuracy_score(y_true, y_pred)

    macro_f1 = float(
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )[2]
    )

    print("\n" + "=" * 60)
    print("MOBILENETV3-SMALL ISIC 2019 EVALUATION COMPLETED")
    print("=" * 60)
    print(f"Evaluated samples: {len(y_true)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    print(f"Classification report: {report_path}")
    print(f"Per-class metrics: {per_class_path}")
    print(f"Confusion matrix: {confusion_path}")
    print("\nClassification report:\n")
    print(report)


# ============================================================
# 14. MAIN
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not IMAGE_DIR.is_dir():
        raise FileNotFoundError(
            f"Image directory not found: {IMAGE_DIR}"
        )

    print("=" * 60)
    print("MOBILENETV3-SMALL ISIC 2019 EVALUATION")
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")
    print(f"Project directory: {PROJECT_DIR}")
    print(f"Image directory: {IMAGE_DIR}")
    print(f"Ground-truth CSV: {GROUND_TRUTH_CSV}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Class order: {CLASS_NAMES}")
    print(f"Device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")

    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load ground-truth labels and keep the seven eligible classes.
    ground_truth = get_ground_truth(GROUND_TRUTH_CSV)

    print(
        "\nEligible ground-truth rows after excluding "
        f"SCC/UNK and invalid labels: {len(ground_truth)}"
    )

    print("\nGround-truth class counts:")
    print(
        ground_truth["label"]
        .value_counts()
        .reindex(CLASS_NAMES, fill_value=0)
    )

    # Load the scaler fitted during centralized training.
    scaler = load_feature_scaler()

    dataset = ISICEvaluationDataset(
        dataframe=ground_truth,
        image_dir=IMAGE_DIR,
        scaler=scaler,
    )

    print(f"\nMatched images: {len(dataset)}")

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )

    print(f"Total batches: {len(dataloader)}")

    print("\nBuilding MobileNetV3-Small hybrid model...")
    model = build_model()

    print("Loading trained MobileNet checkpoint...")
    model = load_checkpoint(model, CHECKPOINT_PATH)

    print("\nStarting external evaluation...")
    print(
        "Handcrafted features are extracted as each sample is loaded."
    )

    y_true, y_pred, image_ids = run_inference(
        model,
        dataloader,
    )

    if len(y_true) != len(dataset):
        raise RuntimeError(
            f"Expected {len(dataset)} predictions, "
            f"received {len(y_true)}."
        )

    prediction_df = pd.DataFrame({
        "image_id": image_ids,
        "true_class": [
            CLASS_NAMES[index] for index in y_true
        ],
        "predicted_class": [
            CLASS_NAMES[index] for index in y_pred
        ],
    })

    prediction_path = OUTPUT_DIR / "predictions.csv"
    prediction_df.to_csv(prediction_path, index=False)

    print(f"\nPer-image predictions saved: {prediction_path}")

    save_evaluation_outputs(y_true, y_pred)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\nEVALUATION FAILED")
        print(f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        raise
