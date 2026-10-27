import os
import importlib.util

import torch
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

NUM_CLASSES = 7
HANDCRAFTED_FEATURE_DIM = 1320

# FP32 = 4 bytes per parameter
BYTES_PER_PARAMETER = 4

OUTPUT_DIR = "outputs"

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "model_parameter_size_comparison.png"
)


# ============================================================
# MODULE LOADER
# ============================================================

def load_module(
    module_path,
    module_name
):
    """
    Dynamically load a Python module from a file path.
    """

    spec = importlib.util.spec_from_file_location(
        module_name,
        module_path
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not load module: {module_path}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


# ============================================================
# PARAMETER COUNT
# ============================================================

def count_parameters(model):
    """
    Count total and trainable parameters.
    """

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return (
        total_parameters,
        trainable_parameters
    )


# ============================================================
# PARAMETER SIZE
# ============================================================

def calculate_parameter_size_mb(
    total_parameters
):
    """
    Calculate FP32 parameter storage size.

    FP32 = 4 bytes per parameter.

    MB is calculated using:
        1 MB = 1024 * 1024 bytes
    """

    total_bytes = (
        total_parameters
        * BYTES_PER_PARAMETER
    )

    total_mb = (
        total_bytes
        / (1024 ** 2)
    )

    return total_mb


# ============================================================
# MODEL ANALYSIS
# ============================================================

def analyze_model(
    model_name,
    model
):
    """
    Analyze total parameter count and FP32
    parameter storage size.
    """

    total_parameters, trainable_parameters = (
        count_parameters(model)
    )

    parameter_size_mb = (
        calculate_parameter_size_mb(
            total_parameters
        )
    )

    trainable_size_mb = (
        calculate_parameter_size_mb(
            trainable_parameters
        )
    )

    print()
    print("=" * 70)

    print(
        f"{model_name}"
    )

    print("=" * 70)

    print(
        f"Total parameters       : "
        f"{total_parameters:,}"
    )

    print(
        f"Trainable parameters   : "
        f"{trainable_parameters:,}"
    )

    print(
        f"FP32 parameter size    : "
        f"{parameter_size_mb:.2f} MB"
    )

    print(
        f"Trainable FP32 size   : "
        f"{trainable_size_mb:.2f} MB"
    )

    return {

        "name":
            model_name,

        "total_parameters":
            total_parameters,

        "trainable_parameters":
            trainable_parameters,

        "parameter_size_mb":
            parameter_size_mb,

        "trainable_size_mb":
            trainable_size_mb
    }


# ============================================================
# MOBILE NET V3-SMALL
# ============================================================

def build_mobilenet():

    model_path = os.path.join(
        "src",
        "07_model.py"
    )

    module = load_module(
        model_path,
        "mobilenet_model"
    )

    if not hasattr(
        module,
        "HybridClassifier"
    ):

        raise AttributeError(
            "HybridClassifier was not found "
            "in src/07_model.py"
        )

    model = module.HybridClassifier(

        num_classes=NUM_CLASSES,

        handcrafted_feature_dim=
            HANDCRAFTED_FEATURE_DIM,

        pretrained=True
    )

    return model


# ============================================================
# CONVNEXT-TINY V2
# ============================================================

def build_convnext():

    possible_paths = [

        os.path.join(
            "src",
            "07A_model_convnext.py"
        ),

        os.path.join(
            "src",
            "07_model_convnext.py"
        ),

        os.path.join(
            "src",
            "07_convnext_model.py"
        )
    ]

    model_path = None

    for path in possible_paths:

        if os.path.exists(path):

            model_path = path

            break

    if model_path is None:

        raise FileNotFoundError(
            "ConvNeXt model file was not found.\n"
            "Expected one of:\n"
            + "\n".join(
                possible_paths
            )
        )

    module = load_module(
        model_path,
        "convnext_model"
    )

    possible_classes = [

        "ConvNeXtHybridClassifier",

        "ConvNeXtHybrid",

        "ConvNeXtClassifier",

        "HybridClassifier",

        "ConvNeXtModel"
    ]

    model_class = None

    for class_name in possible_classes:

        if hasattr(
            module,
            class_name
        ):

            candidate = getattr(
                module,
                class_name
            )

            if isinstance(
                candidate,
                type
            ):

                model_class = candidate

                break

    if model_class is None:

        raise AttributeError(
            "Could not find the ConvNeXt "
            "model class."
        )

    # --------------------------------------------------------
    # Try the normal constructor
    # --------------------------------------------------------

    try:

        model = model_class(

            num_classes=NUM_CLASSES,

            handcrafted_feature_dim=
                HANDCRAFTED_FEATURE_DIM,

            pretrained=True
        )

    except TypeError:

        model = model_class(

            num_classes=NUM_CLASSES,

            handcrafted_feature_dim=
                HANDCRAFTED_FEATURE_DIM
        )

    return model


# ============================================================
# MAXVIT-TINY V2
# ============================================================

def build_maxvit():

    possible_paths = [

        os.path.join(
            "src",
            "07F_model_maxvit_tiny.py"
        ),

        os.path.join(
            "src",
            "07_model_maxvit.py"
        ),

        os.path.join(
            "src",
            "07_maxvit_model.py"
        ),

        os.path.join(
            "src",
            "07_maxvit.py"
        )
    ]

    model_path = None

    for path in possible_paths:

        if os.path.exists(path):

            model_path = path

            break

    if model_path is None:

        raise FileNotFoundError(
            "MaxViT model file was not found.\n"
            "Expected one of:\n"
            + "\n".join(
                possible_paths
            )
        )

    module = load_module(
        model_path,
        "maxvit_model"
    )

    possible_classes = [

        "MaxViTHybridClassifier",

        "MaxViTHybrid",

        "MaxViTClassifier",

        "HybridClassifier",

        "MaxViTModel"
    ]

    model_class = None

    for class_name in possible_classes:

        if hasattr(
            module,
            class_name
        ):

            candidate = getattr(
                module,
                class_name
            )

            if isinstance(
                candidate,
                type
            ):

                model_class = candidate

                break

    if model_class is None:

        raise AttributeError(
            "Could not find the MaxViT "
            "model class."
        )

    # --------------------------------------------------------
    # Try the normal constructor
    # --------------------------------------------------------

    try:

        model = model_class(

            num_classes=NUM_CLASSES,

            handcrafted_feature_dim=
                HANDCRAFTED_FEATURE_DIM,

            pretrained=True
        )

    except TypeError:

        model = model_class(

            num_classes=NUM_CLASSES,

            handcrafted_feature_dim=
                HANDCRAFTED_FEATURE_DIM
        )

    return model


# ============================================================
# BAR CHART
# ============================================================

def create_bar_chart(
    results
):
    """
    Create a bar chart showing FP32 parameter
    storage size in MB.
    """

    model_names = [

        result["name"]

        for result in results
    ]

    parameter_sizes = [

        result["parameter_size_mb"]

        for result in results
    ]

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 6)
    )

    bars = plt.bar(
        model_names,
        parameter_sizes
    )

    # --------------------------------------------------------
    # Axis labels
    # --------------------------------------------------------

    plt.xlabel(
        "Model",
        fontsize=12
    )

    plt.ylabel(
        "Parameter Size (MB)",
        fontsize=12
    )

    plt.title(
        "Model Parameter Size Comparison",
        fontsize=15
    )

    # --------------------------------------------------------
    # Value labels
    # --------------------------------------------------------

    for bar, value in zip(
        bars,
        parameter_sizes
    ):

        plt.text(

            bar.get_x()
            + bar.get_width() / 2,

            bar.get_height(),

            f"{value:.2f} MB",

            ha="center",

            va="bottom",

            fontsize=11
        )

    # --------------------------------------------------------
    # Grid
    # --------------------------------------------------------

    plt.grid(
        axis="y",
        alpha=0.3
    )

    # --------------------------------------------------------
    # Layout
    # --------------------------------------------------------

    plt.tight_layout()

    # --------------------------------------------------------
    # Save high-resolution figure
    # --------------------------------------------------------

    plt.savefig(

        OUTPUT_FILE,

        dpi=300,

        bbox_inches="tight"
    )

    print()
    print("=" * 70)

    print(
        "BAR CHART SAVED"
    )

    print("=" * 70)

    print(
        os.path.abspath(
            OUTPUT_FILE
        )
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    plt.show()


# ============================================================
# SUMMARY TABLE
# ============================================================

def print_summary(
    results
):

    print()
    print()

    print(
        "=" * 100
    )

    print(
        "MODEL PARAMETER SIZE COMPARISON"
    )

    print(
        "=" * 100
    )

    print(

        f"{'Model':<25}"

        f"{'Total Parameters':>20}"

        f"{'Trainable':>20}"

        f"{'FP32 Size (MB)':>20}"
    )

    print(
        "-" * 100
    )

    for result in results:

        print(

            f"{result['name']:<25}"

            f"{result['total_parameters']:>20,}"

            f"{result['trainable_parameters']:>20,}"

            f"{result['parameter_size_mb']:>17.2f} MB"
        )

    print(
        "-" * 100
    )

    print()
    print(
        "Calculation:"
    )

    print(
        "FP32 parameter size = "
        "parameters × 4 bytes / 1024²"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)

    print(
        "FEDERATED SKIN CANCER"
    )

    print(
        "MODEL PARAMETER SIZE COMPARISON"
    )

    print("=" * 70)

    print()
    print(
        "Models:"
    )

    print(
        "  1. MobileNetV3-Small"
    )

    print(
        "  2. ConvNeXt-Tiny V2"
    )

    print(
        "  3. MaxViT-Tiny V2"
    )

    print()
    print(
        f"Number of classes       : "
        f"{NUM_CLASSES}"
    )

    print(
        f"Handcrafted features    : "
        f"{HANDCRAFTED_FEATURE_DIM}"
    )

    print(
        "Parameter precision     : FP32"
    )

    print(
        "Bytes per parameter     : 4"
    )

    results = []

    # ========================================================
    # MOBILE NET
    # ========================================================

    try:

        print()
        print(
            "Loading MobileNetV3-Small..."
        )

        model = build_mobilenet()

        result = analyze_model(

            "MobileNetV3-Small",

            model
        )

        results.append(
            result
        )

        del model

    except Exception as error:

        print()
        print(
            "ERROR - MobileNetV3-Small"
        )

        print(
            error
        )

    # ========================================================
    # CONVNEXT
    # ========================================================

    try:

        print()
        print(
            "Loading ConvNeXt-Tiny V2..."
        )

        model = build_convnext()

        result = analyze_model(

            "ConvNeXt-Tiny V2",

            model
        )

        results.append(
            result
        )

        del model

    except Exception as error:

        print()
        print(
            "ERROR - ConvNeXt-Tiny V2"
        )

        print(
            error
        )

    # ========================================================
    # MAXVIT
    # ========================================================

    try:

        print()
        print(
            "Loading MaxViT-Tiny V2..."
        )

        model = build_maxvit()

        result = analyze_model(

            "MaxViT-Tiny V2",

            model
        )

        results.append(
            result
        )

        del model

    except Exception as error:

        print()
        print(
            "ERROR - MaxViT-Tiny V2"
        )

        print(
            error
        )

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    if len(results) == 0:

        print()
        print(
            "No models were successfully loaded."
        )

        return

    print_summary(
        results
    )

    # ========================================================
    # BAR CHART
    # ========================================================

    if len(results) >= 2:

        create_bar_chart(
            results
        )

    else:

        print()
        print(
            "At least two models are required "
            "for the comparison chart."
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()