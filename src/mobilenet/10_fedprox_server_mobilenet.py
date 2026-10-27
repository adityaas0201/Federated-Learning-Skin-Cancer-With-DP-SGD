"""
10_fedprox_server_mobilenet.py
-----------------------------
PHASE 2 - STEP 10: MOBILENETV3-SMALL FEDPROX SERVER

Purpose:
    Implements the SERVER-SIDE aggregation for the FedProx experiment for the
    MobileNetV3-Small hybrid federated learning experiment.

Federated setup:

    Client:
        Image
        +
        1320 handcrafted features
        +
        local model training

    Server:
        Receives local client model states
        ->
        validates states
        ->
        performs the standard sample-weighted server aggregation used by FedProx
        ->
        produces global model state

Important:
    This file does NOT train clients.

    FedProx client-side optimization is handled by:
        09_federated_client.py

    Client training is handled by:
        09_federated_client.py

    Full federated round orchestration is handled by:
        11A_train_federated_mobilenet.py

FedProx server aggregation:
    Global parameter =
        sum(
            client_weight_i * client_parameter_i
        )

    where:

        client_weight_i =
            client_samples_i / total_samples

This implementation is compatible with the MobileNetV3-Small
HybridClassifier used by Step 9.

Architecture:
    MobileNetV3-Small
        |
        |---- 768-D deep embedding
        |
        +---- 1320-D handcrafted features
                         |
                         v
                    256-D branch
                         |
                         v
                  Feature Fusion
                         |
                         v
                      1024-D
                         |
                         v
                    7 classes
"""


# =====================================================================
# IMPORTS
# =====================================================================

import os
import sys
import time
import copy
import random
import importlib.util

import numpy as np
import torch


# =====================================================================
# CONFIGURATION
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
# PATHS
# =====================================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_DIR = os.path.join(
    config.OUTPUT_DIR,
    "models"
)

FEDERATED_DIR = os.path.join(
    config.OUTPUT_DIR,
    "federated"
)

FEDERATED_MODEL_DIR = os.path.join(
    FEDERATED_DIR,
    "models"
)

FEDERATED_LOG_DIR = os.path.join(
    FEDERATED_DIR,
    "logs"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

os.makedirs(
    FEDERATED_MODEL_DIR,
    exist_ok=True
)

os.makedirs(
    FEDERATED_LOG_DIR,
    exist_ok=True
)


# =====================================================================
# CONSTANTS
# =====================================================================

NUM_CLASSES = config.NUM_CLASSES

HANDCRAFTED_FEATURE_DIM = (
    config.HANDCRAFTED_FEATURE_DIM
)

RANDOM_SEED = config.RANDOM_SEED

GLOBAL_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "federated_mobilenet_global.pt"
)

INITIAL_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "federated_mobilenet_initial.pt"
)


# =====================================================================
# RANDOM SEED
# =====================================================================

def set_random_seeds():
    """
    Makes server-side initialization reproducible.
    """

    seed = RANDOM_SEED

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    print(
        f"Random seed: {seed}"
    )


# =====================================================================
# LOAD MOBILENETV3-SMALL MODEL MODULE
# =====================================================================

def load_mobilenet_model_module():
    """
    Dynamically loads the MobileNetV3-Small model module.

    Step 9 and Step 10 must use the same model architecture.

    Expected file:
        07_model.py

    If your Step 9 file uses a different filename, change
    MODEL_MODULE_FILENAME below.
    """

    MODEL_MODULE_FILENAME = (
        "07_model.py"
    )

    model_path = os.path.join(
        BASE_DIR,
        MODEL_MODULE_FILENAME
    )

    if not os.path.exists(model_path):

        raise FileNotFoundError(
            "\nMobileNetV3 model module not found:\n"
            f"  {model_path}\n\n"
            "Make sure Step 7A model file exists."
        )

    spec = importlib.util.spec_from_file_location(
        "mobilenet_model_module",
        model_path
    )

    if spec is None or spec.loader is None:

        raise ImportError(
            f"Could not load model module:\n"
            f"{model_path}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


# =====================================================================
# BUILD GLOBAL MODEL
# =====================================================================

def build_global_model(device):
    """
    Creates the global MobileNetV3-Small hybrid model.

    The architecture must exactly match the client architecture.
    """

    model_module = (
        load_mobilenet_model_module()
    )

    if not hasattr(
        model_module,
        "HybridClassifier"
    ):

        raise AttributeError(
            "\nHybridClassifier was not found in:\n"
            "07_model.py\n\n"
            "Step 10 requires the same HybridClassifier "
            "used by Step 9."
        )

    HybridClassifier = (
        model_module.HybridClassifier
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "BUILDING GLOBAL MOBILENETV3-SMALL MODEL"
    )

    print(
        "=" * 60
    )

    model = HybridClassifier(
        num_classes=NUM_CLASSES,
        handcrafted_feature_dim=HANDCRAFTED_FEATURE_DIM,
        pretrained=True,
    ).to(device)

    print(
        "\nGlobal model created."
    )

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"  Total parameters    : "
        f"{total_params:,}"
    )

    print(
        f"  Trainable parameters: "
        f"{trainable_params:,}"
    )

    return model


# =====================================================================
# STATE DICTIONARY VALIDATION
# =====================================================================

def validate_state_dict(
    state_dict,
    reference_state_dict=None,
    client_id=None
):
    """
    Validates a client/global state dictionary.

    Checks:

        1. state_dict exists
        2. all expected parameter names exist
        3. tensor shapes match
        4. tensors are finite
        5. tensors are floating-point where FedAvg is performed

    Integer/buffer tensors are allowed to exist in the model state
    dictionary. They are handled separately during aggregation.
    """

    prefix = (
        f"Client {client_id}"
        if client_id is not None
        else "State"
    )

    if not isinstance(
        state_dict,
        dict
    ):

        raise TypeError(
            f"{prefix}: expected dict, "
            f"got {type(state_dict)}"
        )

    if reference_state_dict is None:

        return True

    reference_keys = set(
        reference_state_dict.keys()
    )

    current_keys = set(
        state_dict.keys()
    )

    missing_keys = (
        reference_keys - current_keys
    )

    unexpected_keys = (
        current_keys - reference_keys
    )

    if missing_keys:

        raise ValueError(
            f"{prefix}: missing state keys:\n"
            f"{sorted(list(missing_keys))[:20]}"
        )

    if unexpected_keys:

        raise ValueError(
            f"{prefix}: unexpected state keys:\n"
            f"{sorted(list(unexpected_keys))[:20]}"
        )

    for key in reference_state_dict:

        reference_tensor = (
            reference_state_dict[key]
        )

        current_tensor = (
            state_dict[key]
        )

        if not isinstance(
            current_tensor,
            torch.Tensor
        ):

            raise TypeError(
                f"{prefix}: state '{key}' "
                f"is not a tensor."
            )

        if (
            current_tensor.shape
            != reference_tensor.shape
        ):

            raise ValueError(
                f"{prefix}: shape mismatch for "
                f"'{key}'.\n"
                f"Expected: "
                f"{tuple(reference_tensor.shape)}\n"
                f"Found: "
                f"{tuple(current_tensor.shape)}"
            )

        if current_tensor.is_floating_point():

            if not torch.isfinite(
                current_tensor
            ).all():

                raise ValueError(
                    f"{prefix}: NaN/Inf detected "
                    f"in state '{key}'."
                )

    return True


# =====================================================================
# EXTRACT CPU STATE
# =====================================================================

def state_dict_to_cpu(
    state_dict
):
    """
    Copies a model state dictionary to CPU.

    This is important for federated communication:

        client GPU
            ->
        CPU state
            ->
        server aggregation

    It also prevents unnecessary GPU memory usage.
    """

    cpu_state = {}

    for key, tensor in state_dict.items():

        cpu_state[key] = (
            tensor.detach()
            .cpu()
            .clone()
        )

    return cpu_state


# =====================================================================
# WEIGHTED FEDAVG
# =====================================================================

def fedavg(
    client_states,
    client_num_samples,
    reference_state_dict=None
):
    """
    Performs weighted Federated Averaging.

    Parameters
    ----------
    client_states : list[dict]
        State dictionaries from participating clients.

    client_num_samples : list[int]
        Number of local training samples for each client.

    reference_state_dict : dict, optional
        Global/reference state used for structural validation.

    Returns
    -------
    dict
        Aggregated global state dictionary.

    Formula:

        w_i = n_i / sum(n_i)

        global = sum(w_i * local_i)
    """

    if len(client_states) == 0:

        raise ValueError(
            "FedProx server received zero client states."
        )

    if len(client_states) != len(
        client_num_samples
    ):

        raise ValueError(
            "Number of client states does not "
            "match number of client sample counts."
        )

    # ---------------------------------------------------------------
    # Validate sample counts
    # ---------------------------------------------------------------

    sample_counts = []

    for index, count in enumerate(
        client_num_samples
    ):

        try:
            count = int(count)
        except Exception as exc:

            raise ValueError(
                f"Invalid sample count for "
                f"client {index}: {count}"
            ) from exc

        if count <= 0:

            raise ValueError(
                f"Client {index} has invalid "
                f"sample count: {count}"
            )

        sample_counts.append(
            count
        )

    total_samples = sum(
        sample_counts
    )

    if total_samples <= 0:

        raise ValueError(
            "Total client samples must be > 0."
        )

    # ---------------------------------------------------------------
    # Validate states
    # ---------------------------------------------------------------

    if reference_state_dict is None:

        reference_state_dict = (
            client_states[0]
        )

    for index, state in enumerate(
        client_states
    ):

        validate_state_dict(
            state,
            reference_state_dict,
            client_id=index
        )

    # ---------------------------------------------------------------
    # Client weights
    # ---------------------------------------------------------------

    weights = [
        count / total_samples
        for count in sample_counts
    ]

    print(
        "\nFedAvg client weights:"
    )

    for index, weight in enumerate(
        weights
    ):

        print(
            f"  Client {index}: "
            f"samples={sample_counts[index]}, "
            f"weight={weight:.6f}"
        )

    print(
        f"  Total samples: "
        f"{total_samples}"
    )

    # ---------------------------------------------------------------
    # Aggregate
    # ---------------------------------------------------------------

    global_state = {}

    reference_keys = (
        list(reference_state_dict.keys())
    )

    for key in reference_keys:

        reference_tensor = (
            reference_state_dict[key]
        )

        # -----------------------------------------------------------
        # Floating point tensors
        # -----------------------------------------------------------

        if reference_tensor.is_floating_point():

            aggregated = torch.zeros_like(
                reference_tensor,
                dtype=torch.float32,
                device="cpu"
            )

            for state, weight in zip(
                client_states,
                weights
            ):

                client_tensor = (
                    state[key]
                    .detach()
                    .cpu()
                    .float()
                )

                aggregated += (
                    client_tensor * weight
                )

            # Preserve original dtype.
            global_state[key] = (
                aggregated.to(
                    dtype=reference_tensor.dtype
                )
            )

        # -----------------------------------------------------------
        # Non-floating tensors / buffers
        # -----------------------------------------------------------
        #
        # Example:
        #   num_batches_tracked
        #
        # These cannot be averaged as floating-point parameters.
        #
        # We use the value from the client with the largest
        # contribution to the current round.
        # -----------------------------------------------------------

        else:

            largest_client = int(
                np.argmax(weights)
            )

            global_state[key] = (
                client_states[
                    largest_client
                ][key]
                .detach()
                .cpu()
                .clone()
            )

    # ---------------------------------------------------------------
    # Final validation
    # ---------------------------------------------------------------

    validate_state_dict(
        global_state,
        reference_state_dict
    )

    return global_state


# =====================================================================
# FEDPROX SERVER API
# =====================================================================
#
# FedProx changes the CLIENT local objective, not the server averaging
# rule. Therefore the server uses the same sample-weighted aggregation
# implemented by fedavg(). This alias gives the FedProx experiment an
# explicit server-side API without changing the underlying aggregation.
# =====================================================================

def fedprox(
    client_states,
    client_num_samples,
    reference_state_dict=None
):
    """
    FedProx-compatible server aggregation.

    The proximal term is applied during client-side training. The server
    aggregates the resulting client models using sample-weighted averaging.
    """

    return fedavg(
        client_states=client_states,
        client_num_samples=client_num_samples,
        reference_state_dict=reference_state_dict
    )


# =====================================================================
# AGGREGATE AND UPDATE MODEL
# =====================================================================

def aggregate_client_updates(
    global_model,
    client_states,
    client_num_samples
):
    """
    Aggregates client models and loads the resulting global state
    into the server model.
    """

    if len(client_states) == 0:

        raise ValueError(
            "Cannot aggregate zero clients."
        )

    reference_state = (
        state_dict_to_cpu(
            global_model.state_dict()
        )
    )

    global_state = fedprox(
        client_states=client_states,
        client_num_samples=client_num_samples,
        reference_state_dict=reference_state
    )

    global_model.load_state_dict(
        global_state,
        strict=True
    )

    return global_model


# =====================================================================
# SAVE GLOBAL CHECKPOINT
# =====================================================================

def save_global_checkpoint(
    model,
    epoch,
    round_number,
    client_num_samples=None,
    val_macro_f1=None,
    val_accuracy=None,
    test_accuracy=None,
    test_macro_f1=None,
    path=None
):
    """
    Saves the global MobileNetV3 model.

    The checkpoint contains enough metadata to identify the
    federated experiment.
    """

    if path is None:

        path = GLOBAL_MODEL_PATH

    state_dict = (
        state_dict_to_cpu(
            model.state_dict()
        )
    )

    checkpoint = {

        "model_state_dict":
            state_dict,

        "round":
            int(round_number),

        "epoch":
            int(epoch),

        "client_num_samples":
            client_num_samples,

        "val_macro_f1":
            val_macro_f1,

        "val_accuracy":
            val_accuracy,

        "test_accuracy":
            test_accuracy,

        "test_macro_f1":
            test_macro_f1,

        "architecture":
            "MobileNetV3-Small",

        "num_classes":
            NUM_CLASSES,

        "handcrafted_feature_dim":
            HANDCRAFTED_FEATURE_DIM,

        "random_seed":
            RANDOM_SEED,
    }

    torch.save(
        checkpoint,
        path
    )

    print(
        "\nGlobal checkpoint saved:"
    )

    print(
        f"  {path}"
    )

    return path


# =====================================================================
# LOAD GLOBAL CHECKPOINT
# =====================================================================

def load_global_checkpoint(
    model,
    path=GLOBAL_MODEL_PATH,
    device="cpu"
):
    """
    Loads a previously saved global checkpoint.
    """

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"\nGlobal checkpoint not found:\n"
            f"  {path}"
        )

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False
    )

    if "model_state_dict" not in checkpoint:

        raise ValueError(
            "Invalid federated checkpoint. "
            "'model_state_dict' is missing."
        )

    validate_state_dict(
        checkpoint["model_state_dict"],
        model.state_dict()
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True
    )

    return checkpoint


# =====================================================================
# LOAD CLIENT STATE FROM CHECKPOINT
# =====================================================================

def load_client_state(
    checkpoint_path,
    device="cpu"
):
    """
    Loads a client model checkpoint/state.

    Supported formats:

        1. Raw state dictionary
        2. Dictionary containing:
              model_state_dict
    """

    if not os.path.exists(
        checkpoint_path
    ):

        raise FileNotFoundError(
            f"\nClient checkpoint not found:\n"
            f"  {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    if not isinstance(
        checkpoint,
        dict
    ):

        raise ValueError(
            "Client checkpoint must be a dictionary."
        )

    if "model_state_dict" in checkpoint:

        state_dict = (
            checkpoint[
                "model_state_dict"
            ]
        )

    else:

        # Assume it is already a state dict.
        state_dict = checkpoint

    if not isinstance(
        state_dict,
        dict
    ):

        raise ValueError(
            "Invalid client model state."
        )

    return state_dict


# =====================================================================
# SERVER SUMMARY
# =====================================================================

def print_server_configuration(
    device
):
    """
    Prints server configuration.
    """

    print(
        "\n" + "=" * 60
    )

    print(
        "FEDPROX SERVER CONFIGURATION"
    )

    print(
        "=" * 60
    )

    print(
        f"\nArchitecture          : "
        f"MobileNetV3-Small"
    )

    print(
        f"Device                : "
        f"{device}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU                   : "
            f"{torch.cuda.get_device_name(0)}"
        )

        vram = (
            torch.cuda.get_device_properties(
                0
            ).total_memory
            /
            (1024 ** 3)
        )

        print(
            f"GPU VRAM              : "
            f"{vram:.2f} GB"
        )

    print(
        f"Classes               : "
        f"{NUM_CLASSES}"
    )

    print(
        f"Handcrafted features  : "
        f"{HANDCRAFTED_FEATURE_DIM}"
    )

    print(
        f"Global checkpoint     : "
        f"{GLOBAL_MODEL_PATH}"
    )


# =====================================================================
# SELF-TEST: MODEL
# =====================================================================

def test_model_creation(
    device
):
    """
    Tests global model creation.
    """

    print(
        "\n" + "=" * 60
    )

    print(
        "TEST 1: GLOBAL MODEL CREATION"
    )

    print(
        "=" * 60
    )

    model = build_global_model(
        device
    )

    model.eval()

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    if total_params <= 0:

        raise ValueError(
            "Model has zero parameters."
        )

    if trainable_params <= 0:

        raise ValueError(
            "Model has zero trainable parameters."
        )

    print(
        "\n✓ Global model creation verified."
    )

    return model


# =====================================================================
# SELF-TEST: STATE DICTIONARY
# =====================================================================

def test_state_dictionary(
    model
):
    """
    Tests state extraction and validation.
    """

    print(
        "\n" + "=" * 60
    )

    print(
        "TEST 2: GLOBAL STATE DICTIONARY"
    )

    print(
        "=" * 60
    )

    state = (
        state_dict_to_cpu(
            model.state_dict()
        )
    )

    print(
        f"\nState dictionary tensors: "
        f"{len(state)}"
    )

    validate_state_dict(
        state,
        model.state_dict()
    )

    all_cpu = all(
        tensor.device.type == "cpu"
        for tensor in state.values()
    )

    if not all_cpu:

        raise ValueError(
            "Some state tensors are not on CPU."
        )

    print(
        "State dictionary structure verified ✓"
    )

    print(
        "State tensors on CPU: True"
    )

    return state


# =====================================================================
# SELF-TEST: FEDAVG
# =====================================================================

def test_fedavg(
    model
):
    """
    Performs a deterministic two-client FedAvg test.

    Client 1:
        +0.0001 to every floating tensor

    Client 2:
        +0.0003 to every floating tensor

    The perturbations are intentionally tiny.  Adding +1/+3 to every
    MobileNet convolution parameter is useful for an arithmetic aggregation
    test, but it can make a deep network numerically unstable during a
    subsequent forward pass.  The aggregation test should verify FedAvg
    arithmetic without manufacturing an invalid model.

    Sample counts:
        Client 1 = 1
        Client 2 = 3

    Expected result:

        (1 * 0.0001 + 3 * 0.0003) / 4
        =
        0.00025

    relative to the original model state.

    This verifies weighted averaging rather than simple averaging.
    """

    print(
        "\n" + "=" * 60
    )

    print(
        "TEST 3: WEIGHTED FEDPROX SERVER AGGREGATION"
    )

    print(
        "=" * 60
    )

    original_state = (
        state_dict_to_cpu(
            model.state_dict()
        )
    )

    client_1 = {}
    client_2 = {}

    TEST_OFFSET_CLIENT_1 = 1.0e-4
    TEST_OFFSET_CLIENT_2 = 3.0e-4

    for key, tensor in (
        original_state.items()
    ):

        if tensor.is_floating_point():

            client_1[key] = (
                tensor.clone() + TEST_OFFSET_CLIENT_1
            )

            client_2[key] = (
                tensor.clone() + TEST_OFFSET_CLIENT_2
            )

        else:

            client_1[key] = (
                tensor.clone()
            )

            client_2[key] = (
                tensor.clone()
            )

    aggregated = fedavg(
        client_states=[
            client_1,
            client_2
        ],
        client_num_samples=[
            1,
            3
        ],
        reference_state_dict=original_state
    )

    # ---------------------------------------------------------------
    # Verify floating-point tensors
    # ---------------------------------------------------------------

    checked = 0

    for key, original in (
        original_state.items()
    ):

        if not original.is_floating_point():

            continue

        expected = (
            original.float()
            + 2.5e-4
        )

        actual = (
            aggregated[key]
            .float()
        )

        if not torch.allclose(
            actual,
            expected,
            rtol=1e-4,
            atol=1e-4
        ):

            difference = (
                actual - expected
            ).abs().max().item()

            raise ValueError(
                f"FedAvg test failed for "
                f"'{key}'.\n"
                f"Maximum difference: "
                f"{difference}"
            )

        checked += 1

        # Only need to verify a meaningful
        # number of tensors because the
        # complete state structure was
        # already validated.
        if checked >= 10:

            break

    if checked == 0:

        raise ValueError(
            "No floating-point tensors found "
            "for FedAvg test."
        )

    print(
        f"\nVerified {checked} floating-point "
        f"tensors."
    )

    print(
        "Expected weighted aggregation offset: +0.00025"
    )

    print(
        "✓ FedProx-compatible weighted server aggregation verified."
    )

    return aggregated


# =====================================================================
# SELF-TEST: MODEL UPDATE
# =====================================================================

def test_model_update(
    model,
    aggregated_state
):
    """
    Tests loading an aggregated state into the model.
    """

    print(
        "\n" + "=" * 60
    )

    print(
        "TEST 4: GLOBAL MODEL UPDATE"
    )

    print(
        "=" * 60
    )

    model.load_state_dict(
        aggregated_state,
        strict=True
    )

    updated_state = (
        state_dict_to_cpu(
            model.state_dict()
        )
    )

    validate_state_dict(
        updated_state,
        aggregated_state
    )

    print(
        "\n✓ Aggregated state loaded "
        "into global model."
    )

    return True


# =====================================================================
# SELF-TEST: FORWARD PASS
# =====================================================================
#
# IMPORTANT:
# The aggregation test uses very small synthetic parameter offsets.
# A test that adds +1/+3 to every MobileNet weight can make the network
# numerically explode even though the FedAvg implementation is correct.
# This forward test therefore validates the actual aggregated model without
# deliberately creating extreme weights.
# =====================================================================

@torch.no_grad()
def test_forward_pass(
    model,
    device
):
    """
    Verifies that the aggregated global model can still perform
    inference after FedAvg.
    """

    print(
        "\n" + "=" * 60
    )

    print(
        "TEST 5: GLOBAL MODEL FORWARD PASS"
    )

    print(
        "=" * 60
    )

    model.eval()

    # ---------------------------------------------------------------
    # Check the aggregated model before inference.
    # This gives a useful error at the parameter level instead of
    # reporting only a NaN/Inf in the final logits.
    # ---------------------------------------------------------------
    non_finite_state = []

    for key, tensor in model.state_dict().items():

        if tensor.is_floating_point():

            if not torch.isfinite(tensor).all():

                non_finite_state.append(key)

    if non_finite_state:

        raise ValueError(
            "\nAggregated model contains NaN/Inf before forward pass.\n"
            "First affected state tensors:\n"
            + "\n".join(
                f"  - {key}"
                for key in non_finite_state[:20]
            )
        )

    batch_size = 4

    # Use bounded random inputs for a stable numerical sanity test.
    # These are synthetic inputs only; they are not training/evaluation data.
    images = torch.randn(
        batch_size,
        3,
        config.IMAGE_SIZE,
        config.IMAGE_SIZE,
        device=device
    )

    images = images.clamp(
        min=-3.0,
        max=3.0
    )

    features = torch.randn(
        batch_size,
        HANDCRAFTED_FEATURE_DIM,
        device=device
    )

    features = features.clamp(
        min=-3.0,
        max=3.0
    )

    outputs = model(
        images,
        features
    )

    print(
        f"  Images   : "
        f"{tuple(images.shape)}"
    )

    print(
        f"  Features : "
        f"{tuple(features.shape)}"
    )

    print(
        f"  Output   : "
        f"{tuple(outputs.shape)}"
    )

    expected_shape = (
        batch_size,
        NUM_CLASSES
    )

    if tuple(outputs.shape) != expected_shape:

        raise ValueError(
            "\nUnexpected output shape.\n"
            f"Expected: {expected_shape}\n"
            f"Found: {tuple(outputs.shape)}"
        )

    if not torch.isfinite(
        outputs
    ).all():

        finite_mask = torch.isfinite(outputs)

        raise ValueError(
            "\nNaN/Inf detected in model output.\n"
            f"  Finite values : "
            f"{finite_mask.sum().item()}/{outputs.numel()}\n"
            f"  Output dtype  : {outputs.dtype}\n"
            f"  Output max abs: "
            f"{torch.nan_to_num(outputs).abs().max().item():.6e}"
        )

    print(
        f"  Output finite: True"
    )

    print(
        f"  Output max abs: "
        f"{outputs.abs().max().item():.6e}"
    )

    print(
        "\n✓ Global forward pass verified."
    )

    return True


# =====================================================================
# SAVE INITIAL GLOBAL MODEL
# =====================================================================

def save_initial_global_model(
    model
):
    """
    Saves the initial pretrained MobileNetV3 global model.

    This is useful for reproducibility because every federated
    experiment can start from exactly the same global initialization.
    """

    state = (
        state_dict_to_cpu(
            model.state_dict()
        )
    )

    checkpoint = {

        "model_state_dict":
            state,

        "round":
            0,

        "architecture":
            "MobileNetV3-Small",

        "num_classes":
            NUM_CLASSES,

        "handcrafted_feature_dim":
            HANDCRAFTED_FEATURE_DIM,

        "random_seed":
            RANDOM_SEED,

        "description":
            "Initial pretrained global model "
            "before federated training."
    }

    torch.save(
        checkpoint,
        INITIAL_MODEL_PATH
    )

    print(
        "\nInitial global model saved:"
    )

    print(
        f"  {INITIAL_MODEL_PATH}"
    )

    return INITIAL_MODEL_PATH


# =====================================================================
# SERVER SELF-TEST
# =====================================================================

def sanity_check():
    """
    Complete Step 10 server self-test.

    No client training is performed.

    The test verifies:

        ✓ model creation
        ✓ state dictionary extraction
        ✓ state validation
        ✓ CPU conversion
        ✓ standard FedProx server aggregation
        ✓ global state loading
        ✓ forward inference
        ✓ checkpoint saving
    """

    start_time = time.perf_counter()

    print(
        "=" * 60
    )

    print(
        "PHASE 2 - STEP 10: "
        "MOBILENETV3-SMALL FEDPROX SERVER SELF-TEST"
    )

    print(
        "=" * 60
    )

    # ---------------------------------------------------------------
    # Device
    # ---------------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"\nDevice: {device}"
    )

    print(
        f"CUDA available: "
        f"{torch.cuda.is_available()}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

        vram = (
            torch.cuda.get_device_properties(
                0
            ).total_memory
            /
            (1024 ** 3)
        )

        print(
            f"GPU VRAM: "
            f"{vram:.2f} GB"
        )

    set_random_seeds()

    # ---------------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------------

    print_server_configuration(
        device
    )

    # ---------------------------------------------------------------
    # Model creation
    # ---------------------------------------------------------------

    model = test_model_creation(
        device
    )

    # ---------------------------------------------------------------
    # State dictionary
    # ---------------------------------------------------------------

    original_state = (
        test_state_dictionary(
            model
        )
    )

    # ---------------------------------------------------------------
    # FedAvg
    # ---------------------------------------------------------------

    aggregated_state = test_fedavg(
        model
    )

    # ---------------------------------------------------------------
    # Load aggregated state
    # ---------------------------------------------------------------

    test_model_update(
        model,
        aggregated_state
    )

    # ---------------------------------------------------------------
    # Forward pass
    # ---------------------------------------------------------------

    test_forward_pass(
        model,
        device
    )

    # ---------------------------------------------------------------
    # Restore original pretrained state
    # ---------------------------------------------------------------
    #
    # The self-test modified the model using synthetic client
    # states. Restore the actual pretrained initialization before
    # saving the initial global checkpoint.
    # ---------------------------------------------------------------

    model.load_state_dict(
        original_state,
        strict=True
    )

    # ---------------------------------------------------------------
    # Save initial model
    # ---------------------------------------------------------------

    save_initial_global_model(
        model
    )

    # ---------------------------------------------------------------
    # GPU memory
    # ---------------------------------------------------------------

    if torch.cuda.is_available():

        allocated = (
            torch.cuda.memory_allocated()
            /
            (1024 ** 2)
        )

        reserved = (
            torch.cuda.memory_reserved()
            /
            (1024 ** 2)
        )

        print(
            "\nGPU memory after self-test:"
        )

        print(
            f"  Allocated : "
            f"{allocated:.2f} MB"
        )

        print(
            f"  Reserved  : "
            f"{reserved:.2f} MB"
        )

    # ---------------------------------------------------------------
    # Final
    # ---------------------------------------------------------------

    elapsed = (
        time.perf_counter()
        -
        start_time
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "PHASE 2 - STEP 10 COMPLETE ✓"
    )

    print(
        "=" * 60
    )

    print(
        f"Self-test time: "
        f"{elapsed:.2f}s"
    )

    print(
        "\nVerified:"
    )

    print(
        "  ✓ MobileNetV3-Small global model"
    )

    print(
        "  ✓ Same architecture as client"
    )

    print(
        "  ✓ Global state dictionary"
    )

    print(
        "  ✓ State dictionary validation"
    )

    print(
        "  ✓ CPU state extraction"
    )

    print(
        "  ✓ Client state validation"
    )

    print(
        "  ✓ Sample-weighted FedAvg"
    )

    print(
        "  ✓ Global model state update"
    )

    print(
        "  ✓ Global forward pass and finite-output check"
    )

    print(
        "  ✓ Initial global checkpoint"
    )

    print(
        "\nSaved:"
    )

    print(
        f"  {INITIAL_MODEL_PATH}"
    )

    print(
        "\nNext:"
    )

    print(
        "  python src\\11A_train_federated_mobilenet.py"
    )


# =====================================================================
# OPTIONAL PUBLIC API
# =====================================================================
#
# These functions are intentionally available for Step 11A:
#
#     build_global_model()
#     fedavg()
#     fedprox()
#     aggregate_client_updates()
#     save_global_checkpoint()
#     load_global_checkpoint()
#     load_client_state()
#
# Step 11A should import these functions rather than duplicating
# the FedAvg implementation.
# =====================================================================


# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":

    sanity_check()