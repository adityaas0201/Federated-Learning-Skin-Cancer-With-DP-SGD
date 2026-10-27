"""
STEP 11 — MAIN FEDERATED TRAINING SCRIPT

End-to-end FedAvg training pipeline.

Pipeline:
    1. Initialize global HybridClassifier
    2. For each federated round:
        a. Broadcast global model
        b. Train every client locally using Step 9
        c. Aggregate client models using sample-weighted FedAvg
        d. Evaluate global model on validation set
        e. Save round logs
        f. Save best model based on validation Macro-F1
    3. After all rounds:
        a. Load best validation model
        b. Evaluate once on held-out test set
        c. Save classification report
        d. Save confusion matrix
        e. Save training curves

EXPERIMENT:
    Dataset              : HAM10000
    Classes              : 7
    Clients              : 6
    Dirichlet alpha      : 0.3
    Aggregation          : Sample-weighted FedAvg
    Local epochs         : 3
    Federated rounds     : 15
    Handcrafted features : 1320
    CNN                  : MobileNetV3-Small
    Privacy              : OFF
    DP / noise           : NOT USED IN THIS STEP

IMPORTANT:
    - Validation is used for model selection.
    - Test set is used only once after training.
    - Clients are trained sequentially for RTX 3050 4GB VRAM.
"""

from __future__ import annotations

import csv
import gc
import importlib.util
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


# =====================================================================
# 1. PROJECT ROOT
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =====================================================================
# 2. DYNAMIC MODULE LOADER
# =====================================================================

def load_module(module_name: str, relative_path: str):
    """
    Load a Python module from a path relative to PROJECT_ROOT.
    """

    file_path = PROJECT_ROOT / relative_path

    if not file_path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{file_path}"
        )

    spec = importlib.util.spec_from_file_location(
        module_name,
        file_path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not load module:\n{file_path}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


# =====================================================================
# 3. LOAD PROJECT MODULES
# =====================================================================

print("Loading project modules...")

config = load_module(
    "config",
    "configs/config.py",
)

dataset_module = load_module(
    "dataset_module",
    "src/06_dataset.py",
)

model_module = load_module(
    "model_module",
    "src/07_model.py",
)

client_module = load_module(
    "client_module",
    "src/09_federated_client.py",
)

server_module = load_module(
    "server_module",
    "src/10_fedavg_server.py",
)

print("Project modules loaded successfully.")


# =====================================================================
# 4. OUTPUT DIRECTORIES
# =====================================================================

OUTPUT_DIR = PROJECT_ROOT / "outputs"

MODEL_DIR = OUTPUT_DIR / "models"
LOG_DIR = OUTPUT_DIR / "training_logs"
PLOT_DIR = OUTPUT_DIR / "plots"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

LOG_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PLOT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =====================================================================
# 5. CONFIGURATION
# =====================================================================

NUM_CLIENTS = int(
    config.NUM_CLIENTS
)

NUM_ROUNDS = int(
    config.NUM_FEDERATED_ROUNDS
)

NUM_LOCAL_EPOCHS = int(
    config.NUM_LOCAL_EPOCHS
)

LEARNING_RATE = float(
    config.LEARNING_RATE
)

WEIGHT_DECAY = float(
    config.WEIGHT_DECAY
)

NUM_CLASSES = int(
    config.NUM_CLASSES
)

FEATURE_DIM = int(
    config.HANDCRAFTED_FEATURE_DIM
)

RANDOM_SEED = int(
    getattr(
        config,
        "RANDOM_SEED",
        42,
    )
)

USE_AMP = bool(
    getattr(
        config,
        "USE_AMP",
        True,
    )
)

LABEL_SMOOTHING = float(
    getattr(
        config,
        "LABEL_SMOOTHING",
        0.0,
    )
)


# =====================================================================
# 6. DEVICE
# =====================================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# =====================================================================
# 7. REPRODUCIBILITY
# =====================================================================

def set_seed(seed: int):
    """
    Set random seeds for reproducibility.
    """

    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =====================================================================
# 8. CREATE GLOBAL MODEL
# =====================================================================

def create_global_model():
    """
    Create the initial global HybridClassifier.

    Uses ImageNet-pretrained MobileNetV3-Small,
    matching the centralized baseline.

    IMPORTANT:
    The actual Step 7 constructor uses
    handcrafted_feature_dim.
    """

    model = model_module.HybridClassifier(
        num_classes=NUM_CLASSES,
        handcrafted_feature_dim=FEATURE_DIM,
        pretrained=True,
    )

    return model


# =====================================================================
# 9. CLONE STATE DICTIONARY
# =====================================================================

def clone_state_dict_to_cpu(model):
    """
    Clone model state dictionary to CPU.
    """

    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


# =====================================================================
# 10. LOAD GLOBAL STATE
# =====================================================================

def load_global_state(
    model,
    global_state,
):
    """
    Load global state dictionary into model.
    """

    model.load_state_dict(
        global_state,
        strict=True,
    )


# =====================================================================
# 11. GET CLIENT SIZES
# =====================================================================

def get_client_sizes():
    """
    Get number of training samples for every client.

    Step 6 API:
        build_client_loader(client_id)
    """

    client_sizes = []

    for client_id in range(NUM_CLIENTS):

        loader = dataset_module.build_client_loader(
            client_id
        )

        client_sizes.append(
            len(loader.dataset)
        )

    return client_sizes


# =====================================================================
# 12. CREATE LOG FILES
# =====================================================================

def initialize_log_files():

    round_log_path = (
        LOG_DIR
        / "federated_round_log.csv"
    )

    client_log_path = (
        LOG_DIR
        / "federated_client_log.csv"
    )

    # ---------------------------------------------------------------
    # Round log
    # ---------------------------------------------------------------

    with open(
        round_log_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "round",
            "train_loss_weighted",
            "train_accuracy_weighted",
            "val_loss",
            "val_accuracy",
            "val_macro_f1",
            "learning_rate",
            "round_time_sec",
            "is_best",
            "best_round",
        ])

    # ---------------------------------------------------------------
    # Client log
    # ---------------------------------------------------------------

    with open(
        client_log_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "round",
            "client_id",
            "num_samples",
            "local_epochs",
            "final_local_loss",
            "final_local_accuracy",
            "update_norm",
            "training_time_sec",
        ])

    return (
        round_log_path,
        client_log_path,
    )


# =====================================================================
# 13. APPEND ROUND LOG
# =====================================================================

def append_round_log(
    path,
    round_number,
    train_loss,
    train_accuracy,
    val_loss,
    val_accuracy,
    val_macro_f1,
    learning_rate,
    round_time,
    is_best,
    best_round,
):

    with open(
        path,
        "a",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            round_number,
            f"{train_loss:.6f}",
            f"{train_accuracy:.6f}",
            f"{val_loss:.6f}",
            f"{val_accuracy:.6f}",
            f"{val_macro_f1:.6f}",
            f"{learning_rate:.8f}",
            f"{round_time:.2f}",
            int(is_best),
            best_round,
        ])


# =====================================================================
# 14. APPEND CLIENT LOG
# =====================================================================

def append_client_log(
    path,
    round_number,
    client_id,
    num_samples,
    local_epochs,
    final_loss,
    final_accuracy,
    update_norm,
    training_time,
):

    with open(
        path,
        "a",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            round_number,
            client_id,
            num_samples,
            local_epochs,
            f"{final_loss:.6f}",
            f"{final_accuracy:.6f}",
            f"{update_norm:.6f}",
            f"{training_time:.2f}",
        ])


# =====================================================================
# 15. SAVE CHECKPOINT
# =====================================================================

def save_checkpoint(
    path,
    model,
    round_number,
    val_loss,
    val_accuracy,
    val_macro_f1,
    client_sizes,
):
    """
    Save complete federated checkpoint.
    """

    checkpoint = {

        "round":
            int(round_number),

        "model_state_dict":
            clone_state_dict_to_cpu(
                model
            ),

        "val_loss":
            float(val_loss),

        "val_accuracy":
            float(val_accuracy),

        "val_macro_f1":
            float(val_macro_f1),

        "client_sizes":
            list(client_sizes),

        # Experiment configuration
        "num_clients":
            NUM_CLIENTS,

        "num_rounds":
            NUM_ROUNDS,

        "num_local_epochs":
            NUM_LOCAL_EPOCHS,

        "learning_rate":
            LEARNING_RATE,

        "weight_decay":
            WEIGHT_DECAY,

        "num_classes":
            NUM_CLASSES,

        "handcrafted_feature_dim":
            FEATURE_DIM,

        "random_seed":
            RANDOM_SEED,

        "method":
            "FedAvg",

        "dirichlet_alpha":
            0.3,

        "differential_privacy":
            False,
    }

    torch.save(
        checkpoint,
        path,
    )


# =====================================================================
# 16. PLOT TRAINING CURVES
# =====================================================================

def plot_training_curves(
    round_history,
):

    rounds = [
        item["round"]
        for item in round_history
    ]

    val_accuracy = [
        item["val_accuracy"]
        for item in round_history
    ]

    val_macro_f1 = [
        item["val_macro_f1"]
        for item in round_history
    ]

    val_loss = [
        item["val_loss"]
        for item in round_history
    ]

    # ---------------------------------------------------------------
    # Macro-F1
    # ---------------------------------------------------------------

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        rounds,
        val_macro_f1,
        marker="o",
        label="Validation Macro-F1",
    )

    plt.xlabel("Federated Round")
    plt.ylabel("Macro-F1")
    plt.title("FedAvg Validation Macro-F1")

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        PLOT_DIR
        / "federated_val_macro_f1.png",
        dpi=200,
    )

    plt.close()

    # ---------------------------------------------------------------
    # Accuracy
    # ---------------------------------------------------------------

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        rounds,
        val_accuracy,
        marker="o",
        label="Validation Accuracy",
    )

    plt.xlabel("Federated Round")
    plt.ylabel("Accuracy")
    plt.title("FedAvg Validation Accuracy")

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        PLOT_DIR
        / "federated_val_accuracy.png",
        dpi=200,
    )

    plt.close()

    # ---------------------------------------------------------------
    # Loss
    # ---------------------------------------------------------------

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        rounds,
        val_loss,
        marker="o",
        label="Validation Loss",
    )

    plt.xlabel("Federated Round")
    plt.ylabel("Loss")
    plt.title("FedAvg Validation Loss")

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        PLOT_DIR
        / "federated_val_loss.png",
        dpi=200,
    )

    plt.close()


# =====================================================================
# 17. FINAL TEST EVALUATION
# =====================================================================

def evaluate_test_set(model):
    """
    Evaluate the selected best global model on test data.

    Test data is used only after federated training.
    """

    test_loader = (
        dataset_module.build_eval_loader(
            "test"
        )
    )

    model.eval()

    # Unweighted test loss for final evaluation.
    criterion = torch.nn.CrossEntropyLoss(
        label_smoothing=LABEL_SMOOTHING
    )

    all_labels = []
    all_predictions = []

    total_loss = 0.0
    total_samples = 0

    use_amp_now = (
        USE_AMP
        and DEVICE.type == "cuda"
    )

    with torch.no_grad():

        for (
            images,
            features,
            labels,
        ) in test_loader:

            images = images.to(
                DEVICE,
                non_blocking=True,
            )

            features = features.to(
                DEVICE,
                non_blocking=True,
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True,
            )

            with torch.amp.autocast(
                device_type=DEVICE.type,
                enabled=use_amp_now,
            ):

                logits = model(
                    images,
                    features,
                )

                loss = criterion(
                    logits,
                    labels,
                )

            batch_size = labels.size(0)

            total_loss += (
                loss.item()
                * batch_size
            )

            total_samples += batch_size

            predictions = torch.argmax(
                logits,
                dim=1,
            )

            all_labels.extend(
                labels.cpu().numpy().tolist()
            )

            all_predictions.extend(
                predictions.cpu().numpy().tolist()
            )

    if total_samples == 0:
        raise RuntimeError(
            "Test DataLoader returned zero samples."
        )

    test_loss = (
        total_loss
        / total_samples
    )

    test_accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )

    test_macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    return (
        test_loss,
        test_accuracy,
        test_macro_f1,
        np.array(all_labels),
        np.array(all_predictions),
    )


# =====================================================================
# 18. SAVE CONFUSION MATRIX
# =====================================================================

def save_confusion_matrix(
    labels,
    predictions,
):

    class_names = [
        "nv",
        "mel",
        "bkl",
        "bcc",
        "akiec",
        "vasc",
        "df",
    ]

    cm = confusion_matrix(
        labels,
        predictions,
        labels=list(range(NUM_CLASSES)),
    )

    plt.figure(
        figsize=(8, 7)
    )

    plt.imshow(cm)

    plt.title(
        "FedAvg Test Confusion Matrix"
    )

    plt.xlabel(
        "Predicted Label"
    )

    plt.ylabel(
        "True Label"
    )

    plt.xticks(
        range(NUM_CLASSES),
        class_names,
        rotation=45,
        ha="right",
    )

    plt.yticks(
        range(NUM_CLASSES),
        class_names,
    )

    for i in range(NUM_CLASSES):

        for j in range(NUM_CLASSES):

            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
            )

    plt.colorbar()

    plt.tight_layout()

    plt.savefig(
        PLOT_DIR
        / "federated_confusion_matrix.png",
        dpi=200,
    )

    plt.close()


# =====================================================================
# 19. MAIN
# =====================================================================

def main():

    # -----------------------------------------------------------------
    # Reproducibility
    # -----------------------------------------------------------------

    set_seed(
        RANDOM_SEED
    )

    # -----------------------------------------------------------------
    # Header
    # -----------------------------------------------------------------

    print()

    print("=" * 80)
    print("STEP 11 — MAIN FEDERATED TRAINING")
    print("=" * 80)

    # -----------------------------------------------------------------
    # Configuration
    # -----------------------------------------------------------------

    print()

    print("Configuration")
    print("-" * 80)

    print(
        f"Project root        : {PROJECT_ROOT}"
    )

    print(
        f"Device              : {DEVICE}"
    )

    if DEVICE.type == "cuda":

        print(
            f"GPU                 : "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            f"CUDA version        : "
            f"{torch.version.cuda}"
        )

    print(
        f"Clients             : {NUM_CLIENTS}"
    )

    print(
        f"Federated rounds    : {NUM_ROUNDS}"
    )

    print(
        f"Local epochs        : {NUM_LOCAL_EPOCHS}"
    )

    print(
        f"Learning rate       : {LEARNING_RATE}"
    )

    print(
        f"Weight decay        : {WEIGHT_DECAY}"
    )

    print(
        f"Classes             : {NUM_CLASSES}"
    )

    print(
        f"Handcrafted dim     : {FEATURE_DIM}"
    )

    print(
        f"AMP                 : {USE_AMP}"
    )

    print(
        f"Label smoothing     : {LABEL_SMOOTHING}"
    )

    print(
        f"Random seed         : {RANDOM_SEED}"
    )

    print(
        "Aggregation         : Sample-weighted FedAvg"
    )

    print(
        "Dirichlet alpha     : 0.3"
    )

    print(
        "Differential privacy: OFF"
    )

    # -----------------------------------------------------------------
    # Get client sizes
    # -----------------------------------------------------------------

    print()
    print("Loading client dataset sizes...")

    client_sizes = get_client_sizes()

    total_client_samples = sum(
        client_sizes
    )

    print()
    print("Client training samples:")
    print("-" * 60)

    for client_id, size in enumerate(
        client_sizes
    ):

        percentage = (
            100.0
            * size
            / total_client_samples
        )

        print(
            f"Client {client_id}: "
            f"{size:5d} samples "
            f"({percentage:6.2f}%)"
        )

    print("-" * 60)

    print(
        f"Total: {total_client_samples} samples"
    )

    # -----------------------------------------------------------------
    # Verify expected training size
    # -----------------------------------------------------------------

    if total_client_samples != 8112:

        print()
        print("WARNING:")

        print(
            f"Expected 8112 training samples, "
            f"but found {total_client_samples}."
        )

    else:

        print(
            "\nTraining sample conservation: PASSED"
        )

    # -----------------------------------------------------------------
    # Initialize log files
    # -----------------------------------------------------------------

    (
        round_log_path,
        client_log_path,
    ) = initialize_log_files()

    print()

    print(
        f"Round log : {round_log_path}"
    )

    print(
        f"Client log: {client_log_path}"
    )

    # -----------------------------------------------------------------
    # Validation loader
    # -----------------------------------------------------------------

    print()
    print(
        "Loading validation DataLoader..."
    )

    val_loader = (
        dataset_module.build_eval_loader(
            "val"
        )
    )

    print(
        f"Validation samples: "
        f"{len(val_loader.dataset)}"
    )

    # -----------------------------------------------------------------
    # Initialize global model
    # -----------------------------------------------------------------

    print()
    print(
        "Initializing global model..."
    )

    global_model = create_global_model()

    total_parameters = sum(
        p.numel()
        for p in global_model.parameters()
    )

    print(
        f"Total model parameters: "
        f"{total_parameters:,}"
    )

    global_model = global_model.to(DEVICE)

    # -----------------------------------------------------------------
    # Initial global state
    # -----------------------------------------------------------------

    global_state = clone_state_dict_to_cpu(
        global_model
    )

    print(
        "Global model initialized."
    )

    # -----------------------------------------------------------------
    # IMPORTANT:
    # Build shared criterion ONCE.
    #
    # Step 9 requires criterion as a mandatory argument and provides
    # build_shared_criterion() for the global weighted loss.
    # -----------------------------------------------------------------

    print()
    print(
        "Building globally shared training criterion..."
    )

    criterion = (
        client_module.build_shared_criterion(
            DEVICE
        )
    )

    print(
        "Shared criterion ready."
    )

    # -----------------------------------------------------------------
    # Checkpoint paths
    # -----------------------------------------------------------------

    best_checkpoint_path = (
        MODEL_DIR
        / "federated_fedavg_best.pt"
    )

    final_checkpoint_path = (
        MODEL_DIR
        / "federated_fedavg_final.pt"
    )

    # -----------------------------------------------------------------
    # Training history
    # -----------------------------------------------------------------

    round_history = []

    best_val_macro_f1 = -float("inf")

    best_round = 0

    # -----------------------------------------------------------------
    # Overall timer
    # -----------------------------------------------------------------

    overall_start = time.time()

    # =================================================================
    # FEDERATED TRAINING
    # =================================================================

    print()
    print("=" * 80)
    print("STARTING FEDERATED TRAINING")
    print("=" * 80)

    for round_number in range(
        1,
        NUM_ROUNDS + 1,
    ):

        round_start = time.time()

        print()
        print()
        print("=" * 80)

        print(
            f"FEDERATED ROUND "
            f"{round_number}/{NUM_ROUNDS}"
        )

        print("=" * 80)

        # -------------------------------------------------------------
        # Store local client states
        # -------------------------------------------------------------

        client_states = []

        # -------------------------------------------------------------
        # Weighted training metrics
        # -------------------------------------------------------------

        weighted_loss_sum = 0.0
        weighted_accuracy_sum = 0.0

        # =============================================================
        # CLIENT TRAINING
        # =============================================================

        for client_id in range(
            NUM_CLIENTS
        ):

            print()
            print("-" * 70)

            print(
                f"CLIENT {client_id}"
            )

            print("-" * 70)

            print(
                f"Samples: "
                f"{client_sizes[client_id]}"
            )

            client_start = time.time()

            # ---------------------------------------------------------
            # Step 9 local training
            # ---------------------------------------------------------
            #
            # IMPORTANT:
            # Step 9 signature is:
            #
            # train_client(
            #     client_id,
            #     global_state_dict,
            #     device,
            #     criterion,
            #     num_local_epochs,
            #     learning_rate,
            #     weight_decay,
            #     use_amp
            # )
            #
            # ---------------------------------------------------------

            result = client_module.train_client(
                client_id=client_id,
                global_state_dict=global_state,
                device=DEVICE,
                criterion=criterion,
                num_local_epochs=NUM_LOCAL_EPOCHS,
                learning_rate=LEARNING_RATE,
                weight_decay=WEIGHT_DECAY,
                use_amp=USE_AMP,
            )

            client_time = (
                time.time()
                - client_start
            )

            # ---------------------------------------------------------
            # Validate Step 9 output
            # ---------------------------------------------------------

            if not isinstance(
                result,
                dict,
            ):

                raise TypeError(
                    "Step 9 train_client() "
                    "must return a dictionary."
                )

            if "state_dict" not in result:

                raise KeyError(
                    "Step 9 result is missing "
                    "'state_dict'."
                )

            if "num_samples" not in result:

                raise KeyError(
                    "Step 9 result is missing "
                    "'num_samples'."
                )

            if "client_id" not in result:

                raise KeyError(
                    "Step 9 result is missing "
                    "'client_id'."
                )

            if int(result["client_id"]) != client_id:

                raise ValueError(
                    f"Client result ID mismatch: expected "
                    f"{client_id}, got {result['client_id']}."
                )

            # ---------------------------------------------------------
            # Extract state
            # ---------------------------------------------------------

            client_state = result[
                "state_dict"
            ]

            num_samples = int(
                result["num_samples"]
            )

            # ---------------------------------------------------------
            # Verify sample count
            # ---------------------------------------------------------

            expected_samples = (
                client_sizes[client_id]
            )

            if (
                num_samples
                != expected_samples
            ):

                raise ValueError(
                    f"Client {client_id}: "
                    f"Step 9 returned "
                    f"{num_samples} samples, "
                    f"but expected "
                    f"{expected_samples}."
                )

            # ---------------------------------------------------------
            # Store client model
            # ---------------------------------------------------------

            # IMPORTANT:
            # Step 10 expects the COMPLETE Step 9 client-result dictionary,
            # not only result["state_dict"].
            #
            # The result contains:
            #   client_id
            #   state_dict
            #   num_samples
            #   local metrics
            #   parameter_update_norm
            #
            # FedAvg uses client_id and num_samples for validation and
            # sample-weighted aggregation.
            client_states.append(
                result
            )

            # ---------------------------------------------------------
            # IMPORTANT:
            # Use the ACTUAL Step 9 return keys.
            #
            # Step 9 returns:
            #   final_train_loss
            #   final_train_acc
            #   parameter_update_norm
            # ---------------------------------------------------------

            final_loss = float(
                result["final_train_loss"]
            )

            final_accuracy = float(
                result["final_train_acc"]
            )

            update_norm = float(
                result["parameter_update_norm"]
            )

            # ---------------------------------------------------------
            # Weighted local training metrics
            # ---------------------------------------------------------

            weighted_loss_sum += (
                final_loss
                * num_samples
            )

            weighted_accuracy_sum += (
                final_accuracy
                * num_samples
            )

            # ---------------------------------------------------------
            # Client log
            # ---------------------------------------------------------

            append_client_log(
                client_log_path,
                round_number,
                client_id,
                num_samples,
                NUM_LOCAL_EPOCHS,
                final_loss,
                final_accuracy,
                update_norm,
                client_time,
            )

            print()
            print(
                f"Client {client_id} finished."
            )

            print(
                f"Final local loss: "
                f"{final_loss:.4f}"
            )

            print(
                f"Final local accuracy: "
                f"{final_accuracy:.4f}"
            )

            print(
                f"Update norm: "
                f"{update_norm:.6f}"
            )

            print(
                f"Training time: "
                f"{client_time / 60:.2f} min"
            )

            # ---------------------------------------------------------
            # GPU cleanup
            # ---------------------------------------------------------

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # =============================================================
        # WEIGHTED TRAINING METRICS
        # =============================================================

        weighted_train_loss = (
            weighted_loss_sum
            / total_client_samples
        )

        weighted_train_accuracy = (
            weighted_accuracy_sum
            / total_client_samples
        )

        # =============================================================
        # FEDAVG SERVER AGGREGATION
        # =============================================================

        print()
        print("=" * 70)

        print(
            "SERVER — FEDAVG AGGREGATION"
        )

        print("=" * 70)

        aggregation_start = time.time()

        global_state,fedavg_weights,total_samples = (
            server_module.fedavg_aggregate(
                client_states,
            )
        )

        aggregation_time = (
            time.time()
            - aggregation_start
        )

        print(
            f"FedAvg aggregation complete "
            f"in {aggregation_time:.2f}s"
        )

        # =============================================================
        # LOAD AGGREGATED GLOBAL MODEL
        # =============================================================

        load_global_state(
            global_model,
            global_state,
        )

        # =============================================================
        # VALIDATION
        # =============================================================

        print()
        print("=" * 70)

        print(
            "GLOBAL MODEL — VALIDATION"
        )

        print("=" * 70)

        global_model.eval()

        # -------------------------------------------------------------
        # Use the SAME shared criterion used during local training.
        # -------------------------------------------------------------

        (
    val_loss,
    val_accuracy,
    val_macro_f1,
    _,
    _,
) = client_module.evaluate(
    global_model,
    val_loader,
    DEVICE,
    criterion,
)

        # =============================================================
        # ROUND TIMING
        # =============================================================

        round_time = (
            time.time()
            - round_start
        )

        # =============================================================
        # BEST MODEL SELECTION
        # =============================================================

        is_best = (
            val_macro_f1
            > best_val_macro_f1
        )

        if is_best:

            best_val_macro_f1 = (
                val_macro_f1
            )

            best_round = (
                round_number
            )

            # ---------------------------------------------------------
            # Save best checkpoint
            # ---------------------------------------------------------

            save_checkpoint(
                best_checkpoint_path,
                global_model,
                round_number,
                val_loss,
                val_accuracy,
                val_macro_f1,
                client_sizes,
            )

            print()
            print(
                "*** NEW BEST GLOBAL MODEL ***"
            )

            print(
                "Saved to:"
            )

            print(
                f"  {best_checkpoint_path}"
            )

        # =============================================================
        # STORE ROUND HISTORY
        # =============================================================

        round_history.append({

            "round":
                round_number,

            "train_loss":
                weighted_train_loss,

            "train_accuracy":
                weighted_train_accuracy,

            "val_loss":
                float(val_loss),

            "val_accuracy":
                float(val_accuracy),

            "val_macro_f1":
                float(val_macro_f1),
        })

        # =============================================================
        # SAVE ROUND LOG
        # =============================================================

        append_round_log(
            round_log_path,
            round_number,
            weighted_train_loss,
            weighted_train_accuracy,
            val_loss,
            val_accuracy,
            val_macro_f1,
            LEARNING_RATE,
            round_time,
            is_best,
            best_round,
        )

        # =============================================================
        # ROUND SUMMARY
        # =============================================================

        print()
        print("=" * 80)

        print(
            f"ROUND {round_number} SUMMARY"
        )

        print("=" * 80)

        print(
            f"Weighted train loss     : "
            f"{weighted_train_loss:.4f}"
        )

        print(
            f"Weighted train accuracy : "
            f"{weighted_train_accuracy:.4f}"
        )

        print(
            f"Validation loss         : "
            f"{val_loss:.4f}"
        )

        print(
            f"Validation accuracy     : "
            f"{val_accuracy:.4f}"
        )

        print(
            f"Validation Macro-F1     : "
            f"{val_macro_f1:.4f}"
        )

        print(
            f"Best validation Macro-F1: "
            f"{best_val_macro_f1:.4f}"
        )

        print(
            f"Best round              : "
            f"{best_round}"
        )

        print(
            f"Round time              : "
            f"{round_time / 60:.2f} min"
        )

        if is_best:

            print(
                "Status                  : NEW BEST"
            )

        else:

            print(
                "Status                  : "
                "No improvement"
            )

        # -------------------------------------------------------------
        # GPU memory
        # -------------------------------------------------------------

        if torch.cuda.is_available():

            allocated = (
                torch.cuda.memory_allocated()
                / (1024 ** 3)
            )

            reserved = (
                torch.cuda.memory_reserved()
                / (1024 ** 3)
            )

            print(
                f"GPU allocated           : "
                f"{allocated:.2f} GB"
            )

            print(
                f"GPU reserved            : "
                f"{reserved:.2f} GB"
            )

        # -------------------------------------------------------------
        # Cleanup
        # -------------------------------------------------------------

        del client_states

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # =================================================================
    # SAVE FINAL GLOBAL MODEL
    # =================================================================

    print()
    print("=" * 80)

    print(
        "SAVING FINAL GLOBAL MODEL"
    )

    print("=" * 80)

    final_round_metrics = round_history[-1]

    save_checkpoint(
        final_checkpoint_path,
        global_model,
        NUM_ROUNDS,
        final_round_metrics["val_loss"],
        final_round_metrics["val_accuracy"],
        final_round_metrics["val_macro_f1"],
        client_sizes,
    )

    print(
        f"Final model saved to:"
    )

    print(
        f"  {final_checkpoint_path}"
    )

    # =================================================================
    # SAVE TRAINING CURVES
    # =================================================================

    print()
    print(
        "Saving federated training curves..."
    )

    plot_training_curves(
        round_history
    )

    print(
        f"Training curves saved to:"
    )

    print(
        f"  {PLOT_DIR}"
    )

    # =================================================================
    # LOAD BEST MODEL
    # =================================================================

    print()
    print("=" * 80)

    print(
        "LOADING BEST FEDERATED MODEL"
    )

    print("=" * 80)

    best_checkpoint = torch.load(
        best_checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    global_model.load_state_dict(
        best_checkpoint[
            "model_state_dict"
        ],
        strict=True,
    )

    global_model = global_model.to(
        DEVICE
    )

    print(
        f"Best round: "
        f"{best_checkpoint['round']}"
    )

    print(
        f"Best validation loss: "
        f"{best_checkpoint['val_loss']:.4f}"
    )

    print(
        f"Best validation accuracy: "
        f"{best_checkpoint['val_accuracy']:.4f}"
    )

    print(
        f"Best validation Macro-F1: "
        f"{best_checkpoint['val_macro_f1']:.4f}"
    )

    # =================================================================
    # FINAL TEST EVALUATION
    # =================================================================

    print()
    print("=" * 80)

    print(
        "FINAL TEST EVALUATION"
    )

    print("=" * 80)

    (
        test_loss,
        test_accuracy,
        test_macro_f1,
        test_labels,
        test_predictions,
    ) = evaluate_test_set(
        global_model
    )

    print()

    print(
        f"Test Loss     : "
        f"{test_loss:.4f}"
    )

    print(
        f"Test Accuracy : "
        f"{test_accuracy:.4f}"
    )

    print(
        f"Test Macro-F1 : "
        f"{test_macro_f1:.4f}"
    )

    # =================================================================
    # CLASSIFICATION REPORT
    # =================================================================

    class_names = [
        "nv",
        "mel",
        "bkl",
        "bcc",
        "akiec",
        "vasc",
        "df",
    ]

    report = classification_report(
        test_labels,
        test_predictions,
        labels=list(
            range(NUM_CLASSES)
        ),
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    print()
    print("=" * 80)

    print(
        "CLASSIFICATION REPORT"
    )

    print("=" * 80)

    print(report)

    # -----------------------------------------------------------------
    # Save classification report
    # -----------------------------------------------------------------

    report_path = (
        LOG_DIR
        / "federated_test_classification_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "FedAvg Final Test Evaluation\n"
        )

        f.write(
            "=" * 80
            + "\n"
        )

        f.write(
            f"Best round: "
            f"{best_checkpoint['round']}\n"
        )

        f.write(
            f"Best validation Macro-F1: "
            f"{best_checkpoint['val_macro_f1']:.6f}\n"
        )

        f.write(
            f"Test Loss: "
            f"{test_loss:.6f}\n"
        )

        f.write(
            f"Test Accuracy: "
            f"{test_accuracy:.6f}\n"
        )

        f.write(
            f"Test Macro-F1: "
            f"{test_macro_f1:.6f}\n"
        )

        f.write(
            "\nClassification Report\n"
        )

        f.write(
            "-" * 80
            + "\n"
        )

        f.write(report)

    print()
    print(
        "Classification report saved to:"
    )

    print(
        f"  {report_path}"
    )

    # =================================================================
    # CONFUSION MATRIX
    # =================================================================

    save_confusion_matrix(
        test_labels,
        test_predictions,
    )

    print(
        "Confusion matrix saved to:"
    )

    print(
        f"  "
        f"{PLOT_DIR / 'federated_confusion_matrix.png'}"
    )

    # =================================================================
    # TOTAL TIME
    # =================================================================

    total_time = (
        time.time()
        - overall_start
    )

    # =================================================================
    # FINAL SUMMARY
    # =================================================================

    print()
    print()
    print("=" * 80)

    print(
        "STEP 11 COMPLETE"
    )

    print("=" * 80)

    print()

    print(
        f"Best federated round : "
        f"{best_round}"
    )

    print(
        f"Best validation F1   : "
        f"{best_val_macro_f1:.4f}"
    )

    print(
        f"Final test loss      : "
        f"{test_loss:.4f}"
    )

    print(
        f"Final test accuracy  : "
        f"{test_accuracy:.4f}"
    )

    print(
        f"Final test Macro-F1  : "
        f"{test_macro_f1:.4f}"
    )

    print()

    print(
        f"Total training time  : "
        f"{total_time / 60:.2f} minutes"
    )

    print()

    print(
        "Saved files:"
    )

    print("-" * 80)

    print(
        "Best model:"
    )

    print(
        f"  {best_checkpoint_path}"
    )

    print()

    print(
        "Final model:"
    )

    print(
        f"  {final_checkpoint_path}"
    )

    print()

    print(
        "Round log:"
    )

    print(
        f"  {round_log_path}"
    )

    print()

    print(
        "Client log:"
    )

    print(
        f"  {client_log_path}"
    )

    print()

    print(
        "Test report:"
    )

    print(
        f"  {report_path}"
    )

    print()

    print(
        "Plots:"
    )

    print(
        f"  {PLOT_DIR}"
    )

    print()

    print("=" * 80)

    print(
        "FEDAVG EXPERIMENT FINISHED"
    )

    print("=" * 80)


# =====================================================================
# 20. ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    main()