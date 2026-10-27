import os
import importlib.util
import gc

import torch
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

NUM_CLASSES = 7
HANDCRAFTED_FEATURE_DIM = 1320
BYTES_PER_PARAMETER = 4  # FP32 parameter storage estimate
OUTPUT_DIR = "outputs"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "model_parameter_count_comparison.png",
)


# ============================================================
# MODULE LOADER
# ============================================================

def load_module(module_path, module_name):
    """Dynamically load a Python module from a file path."""
    if not os.path.exists(module_path):
        raise FileNotFoundError(f"Model file not found: {module_path}")

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module: {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ============================================================
# PARAMETER COUNTING
# ============================================================

def count_parameters(model):
    """Return total and trainable parameter counts."""
    total_parameters = sum(p.numel() for p in model.parameters())
    trainable_parameters = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    return total_parameters, trainable_parameters


def calculate_parameter_size_mb(parameter_count):
    """Estimate FP32 storage for parameters only, in MiB (1024-based)."""
    return parameter_count * BYTES_PER_PARAMETER / (1024 ** 2)


def analyze_model(model_name, model):
    """Collect parameter statistics for one model."""
    total_parameters, trainable_parameters = count_parameters(model)
    result = {
        "name": model_name,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "parameter_size_mb": calculate_parameter_size_mb(total_parameters),
        "trainable_size_mb": calculate_parameter_size_mb(trainable_parameters),
    }

    print(f"\n{'=' * 68}")
    print(model_name)
    print(f"{'=' * 68}")
    print(f"Total parameters       : {total_parameters:,}")
    print(f"Trainable parameters   : {trainable_parameters:,}")
    print(f"FP32 parameter size    : {result['parameter_size_mb']:.2f} MiB")
    print(f"Trainable FP32 size    : {result['trainable_size_mb']:.2f} MiB")
    return result


# ============================================================
# MOBILENETV3-SMALL
# ============================================================

def build_mobilenet():
    module = load_module(
        os.path.join("src", "07_model.py"),
        "mobilenet_model",
    )

    if not hasattr(module, "HybridClassifier"):
        raise AttributeError(
            "HybridClassifier was not found in src/07_model.py"
        )

    return module.HybridClassifier(
        num_classes=NUM_CLASSES,
        handcrafted_feature_dim=HANDCRAFTED_FEATURE_DIM,
        pretrained=True,
    )


# ============================================================
# CONVNEXT-TINY V2
# ============================================================

def build_convnext():
    possible_paths = [
        os.path.join("src", "07A_model_convnext.py"),
        os.path.join("src", "07_model_convnext.py"),
        os.path.join("src", "07_convnext_model.py"),
    ]

    model_path = next((path for path in possible_paths if os.path.exists(path)), None)
    if model_path is None:
        raise FileNotFoundError(
            "ConvNeXt model file was not found. Expected one of:\n"
            + "\n".join(possible_paths)
        )

    module = load_module(model_path, "convnext_model")
    possible_classes = [
        "ConvNeXtHybridClassifier",
        "ConvNeXtHybrid",
        "ConvNeXtClassifier",
        "HybridClassifier",
        "ConvNeXtModel",
    ]
    model_class = next(
        (getattr(module, name) for name in possible_classes
         if isinstance(getattr(module, name, None), type)),
        None,
    )
    if model_class is None:
        raise AttributeError("Could not find the ConvNeXt model class.")

    try:
        return model_class(
            num_classes=NUM_CLASSES,
            handcrafted_feature_dim=HANDCRAFTED_FEATURE_DIM,
            pretrained=True,
        )
    except TypeError:
        return model_class(
            num_classes=NUM_CLASSES,
            handcrafted_feature_dim=HANDCRAFTED_FEATURE_DIM,
        )


# ============================================================
# MAXVIT-TINY V2
# ============================================================

def build_maxvit():
    possible_paths = [
        os.path.join("src", "07F_model_maxvit_tiny.py"),
        os.path.join("src", "07_model_maxvit.py"),
        os.path.join("src", "07_maxvit_model.py"),
        os.path.join("src", "07_maxvit.py"),
    ]

    model_path = next((path for path in possible_paths if os.path.exists(path)), None)
    if model_path is None:
        raise FileNotFoundError(
            "MaxViT model file was not found. Expected one of:\n"
            + "\n".join(possible_paths)
        )

    module = load_module(model_path, "maxvit_model")
    possible_classes = [
        "MaxViTHybridClassifier",
        "MaxViTHybrid",
        "MaxViTClassifier",
        "HybridClassifier",
        "MaxViTModel",
    ]
    model_class = next(
        (getattr(module, name) for name in possible_classes
         if isinstance(getattr(module, name, None), type)),
        None,
    )
    if model_class is None:
        raise AttributeError("Could not find the MaxViT model class.")

    try:
        return model_class(
            num_classes=NUM_CLASSES,
            handcrafted_feature_dim=HANDCRAFTED_FEATURE_DIM,
            pretrained=True,
        )
    except TypeError:
        return model_class(
            num_classes=NUM_CLASSES,
            handcrafted_feature_dim=HANDCRAFTED_FEATURE_DIM,
        )


# ============================================================
# PARAMETER COUNT BAR CHART
# ============================================================

def create_bar_chart(results):
    """Plot total parameter counts in millions and label each bar."""
    model_names = [result["name"] for result in results]
    parameter_counts = [result["total_parameters"] for result in results]
    parameter_counts_millions = [count / 1_000_000 for count in parameter_counts]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, parameter_counts_millions, width=0.55)

    plt.xlabel("Model", fontsize=12)
    plt.ylabel("Number of Parameters (Millions)", fontsize=12)
    plt.title("Model Parameter Count Comparison", fontsize=15)

    for bar, count, millions in zip(
        bars, parameter_counts, parameter_counts_millions
    ):
        plt.annotate(
            f"{count:,}\n({millions:.2f}M)",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.grid(axis="y", alpha=0.3)
    plt.margins(y=0.15)
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
    print(f"\nParameter comparison plot saved to: {os.path.abspath(OUTPUT_FILE)}")
    plt.show()
    plt.close()


# ============================================================
# SUMMARY TABLE
# ============================================================

def print_summary(results):
    print("\n" + "=" * 112)
    print("MODEL PARAMETER COUNT COMPARISON")
    print("=" * 112)
    print(
        f"{'Model':<25}"
        f"{'Total Parameters':>22}"
        f"{'Trainable Parameters':>25}"
        f"{'FP32 Size (MiB)':>20}"
    )
    print("-" * 112)

    for result in results:
        print(
            f"{result['name']:<25}"
            f"{result['total_parameters']:>22,}"
            f"{result['trainable_parameters']:>25,}"
            f"{result['parameter_size_mb']:>17.2f} MiB"
        )

    print("-" * 112)
    print("Chart metric: total parameter count (millions).")
    print("FP32 size is an estimate for parameter storage only, not total GPU memory.")


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "=" * 68)
    print("FEDERATED SKIN CANCER - MODEL PARAMETER COMPARISON")
    print("=" * 68)
    print("Models: MobileNetV3-Small, ConvNeXt-Tiny V2, MaxViT-Tiny V2")
    print(f"Number of classes      : {NUM_CLASSES}")
    print(f"Handcrafted features   : {HANDCRAFTED_FEATURE_DIM}")
    print("Parameter precision    : FP32 for size estimate")

    model_builders = [
        ("MobileNetV3-Small", build_mobilenet),
        ("ConvNeXt-Tiny V2", build_convnext),
        ("MaxViT-Tiny V2", build_maxvit),
    ]
    results = []

    for model_name, builder in model_builders:
        try:
            print(f"\nLoading {model_name}...")
            model = builder()
            results.append(analyze_model(model_name, model))
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as error:
            print(f"\nERROR - {model_name}: {error}")

    if not results:
        print("\nNo models were successfully loaded. Check model paths and class names.")
        return

    print_summary(results)

    if len(results) >= 2:
        create_bar_chart(results)
    else:
        print("\nAt least two models must load successfully to create a comparison chart.")


if __name__ == "__main__":
    main()
