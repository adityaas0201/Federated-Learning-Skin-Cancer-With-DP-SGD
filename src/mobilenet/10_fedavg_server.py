"""
10_fedavg_server.py
===================

PHASE 2 - STEP 10
FEDAVG / FEDPROX SERVER-SIDE LOGIC

Runs after:
    06_dataset.py
    07_model.py
    08_train_centralized.py
    09_federated_client.py

PURPOSE
-------
This module implements the server-side aggregation used by the
federated learning system.

IMPORTANT SCIENTIFIC DISTINCTION
--------------------------------
FedAvg and FedProx use the SAME server-side aggregation:

    W_global = sum_k (n_k / N) * W_k

The difference is:

    FedAvg:
        Standard local training on each client.

    FedProx:
        Local client objective includes an additional proximal term:

        L_prox = L_local + (mu / 2) * ||w - w_global||^2

Therefore:

    Step 10:
        Implements sample-weighted aggregation.

    Step 9 / later client implementation:
        Will implement the FedProx local proximal objective.

Differential Privacy is NOT implemented here.
It will be added later as a separate privacy experiment.
"""

# =====================================================================
# IMPORTS
# =====================================================================

import os
import sys
import time
import copy
import importlib.util

import torch


# =====================================================================
# CONFIG
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
# LOAD MODEL MODULE
# =====================================================================

def _load_module(filename, module_name):
    """
    Dynamically load a Python module from the src directory.
    """

    path = os.path.join(
        os.path.dirname(__file__),
        filename
    )

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required module not found:\n{path}"
        )

    spec = importlib.util.spec_from_file_location(
        module_name,
        path
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not load module:\n{path}"
        )

    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    return module


model_module = _load_module(
    "07_model.py",
    "model_module"
)

HybridClassifier = model_module.HybridClassifier


# =====================================================================
# CONFIGURATION VALIDATION
# =====================================================================

def validate_config():
    """
    Validate configuration required by the server.
    """

    required = [
        "NUM_CLIENTS",
        "NUM_CLASSES",
        "HANDCRAFTED_FEATURE_DIM",
    ]

    missing = [
        name
        for name in required
        if not hasattr(config, name)
    ]

    if missing:
        raise AttributeError(
            "Missing configuration values:\n"
            + "\n".join(
                f"  - {name}"
                for name in missing
            )
        )

    if config.NUM_CLIENTS <= 0:
        raise ValueError(
            "NUM_CLIENTS must be greater than zero."
        )

    if config.NUM_CLASSES <= 1:
        raise ValueError(
            "NUM_CLASSES must be greater than one."
        )

    if config.HANDCRAFTED_FEATURE_DIM <= 0:
        raise ValueError(
            "HANDCRAFTED_FEATURE_DIM must be greater than zero."
        )


# =====================================================================
# STATE DICTIONARY VALIDATION
# =====================================================================

def verify_state_dict_compatibility(
    reference_state_dict,
    candidate_state_dict,
    client_id=None
):
    """
    Verify that candidate and reference state dictionaries have:

        - identical keys
        - identical tensor shapes

    This is essential before FedAvg.
    """

    prefix = ""

    if client_id is not None:
        prefix = f"Client {client_id}: "

    reference_keys = set(
        reference_state_dict.keys()
    )

    candidate_keys = set(
        candidate_state_dict.keys()
    )

    if reference_keys != candidate_keys:

        missing = reference_keys - candidate_keys
        extra = candidate_keys - reference_keys

        raise RuntimeError(
            f"{prefix}state_dict keys do not match.\n"
            f"Missing: {missing}\n"
            f"Extra: {extra}"
        )

    for key in reference_state_dict:

        expected_shape = (
            reference_state_dict[key].shape
        )

        actual_shape = (
            candidate_state_dict[key].shape
        )

        if expected_shape != actual_shape:

            raise RuntimeError(
                f"{prefix}Shape mismatch for '{key}'. "
                f"Expected {expected_shape}, "
                f"got {actual_shape}."
            )


# =====================================================================
# CALCULATE CLIENT WEIGHTS
# =====================================================================

def calculate_fedavg_weights(client_results):
    """
    Calculate sample-size-weighted FedAvg contribution.

    Formula:

        weight_k = n_k / sum(n_j)

    Parameters
    ----------
    client_results:
        List of dictionaries returned by train_client().

    Returns
    -------
    dict
        client_id -> normalized aggregation weight
    """

    if not client_results:
        raise ValueError(
            "No client results were provided."
        )

    sample_counts = {}

    for result in client_results:

        if "client_id" not in result:
            raise KeyError(
                "Client result is missing 'client_id'."
            )

        if "num_samples" not in result:
            raise KeyError(
                f"Client {result['client_id']} "
                "is missing 'num_samples'."
            )

        client_id = result["client_id"]
        num_samples = result["num_samples"]

        if not isinstance(num_samples, int):
            raise TypeError(
                f"Client {client_id}: num_samples "
                f"must be int, got {type(num_samples)}."
            )

        if num_samples <= 0:
            raise ValueError(
                f"Client {client_id}: invalid sample count "
                f"{num_samples}."
            )

        if client_id in sample_counts:
            raise ValueError(
                f"Duplicate client ID: {client_id}"
            )

        sample_counts[client_id] = num_samples

    total_samples = sum(
        sample_counts.values()
    )

    if total_samples <= 0:
        raise ValueError(
            "Total samples must be greater than zero."
        )

    weights = {
        client_id:
            num_samples / total_samples
        for client_id, num_samples
        in sample_counts.items()
    }

    # Final numerical sanity check.
    weight_sum = sum(
        weights.values()
    )

    if abs(weight_sum - 1.0) > 1e-8:
        raise RuntimeError(
            f"FedAvg weights do not sum to 1. "
            f"Got {weight_sum}."
        )

    return weights


# =====================================================================
# FEDAVG AGGREGATION
# =====================================================================

def fedavg_aggregate(
    client_results,
    reference_state_dict=None
):
    """
    Perform standard sample-weighted FedAvg.

    Formula:

        W_global =
            sum_k (n_k / N) * W_k

    Parameters
    ----------
    client_results:
        List containing local model state_dicts and sample counts.

    reference_state_dict:
        Optional current global model state_dict.

        Used for:
            - key/shape validation
            - handling non-floating state tensors

    Returns
    -------
    aggregated_state_dict
    fedavg_weights
    total_samples
    """

    if not client_results:
        raise ValueError(
            "Cannot aggregate zero clients."
        )

    # -----------------------------------------------------------------
    # Calculate sample weights
    # -----------------------------------------------------------------

    fedavg_weights = calculate_fedavg_weights(
        client_results
    )

    total_samples = sum(
        result["num_samples"]
        for result in client_results
    )

    # -----------------------------------------------------------------
    # Reference model
    # -----------------------------------------------------------------

    if reference_state_dict is None:

        reference_state_dict = (
            client_results[0]["state_dict"]
        )

    # -----------------------------------------------------------------
    # Validate every client
    # -----------------------------------------------------------------

    for result in client_results:

        verify_state_dict_compatibility(
            reference_state_dict,
            result["state_dict"],
            result["client_id"]
        )

    # -----------------------------------------------------------------
    # Aggregate
    # -----------------------------------------------------------------

    aggregated_state_dict = {}

    for key in reference_state_dict:

        reference_tensor = (
            reference_state_dict[key]
        )

        # =============================================================
        # FLOATING-POINT PARAMETERS
        # =============================================================

        if reference_tensor.is_floating_point():

            aggregated_tensor = torch.zeros_like(
                reference_tensor,
                dtype=torch.float32
            )

            for result in client_results:

                client_id = result["client_id"]

                client_weight = (
                    fedavg_weights[client_id]
                )

                client_tensor = (
                    result["state_dict"][key]
                    .to(dtype=torch.float32)
                )

                aggregated_tensor += (
                    client_weight * client_tensor
                )

            # Preserve original dtype.
            aggregated_state_dict[key] = (
                aggregated_tensor.to(
                    dtype=reference_tensor.dtype
                )
            )

        # =============================================================
        # NON-FLOATING STATE
        # =============================================================
        #
        # Example:
        #     BatchNorm num_batches_tracked
        #
        # Such tensors should not be averaged as floating-point
        # parameters.
        #
        # We retain the reference global value.
        # =============================================================

        else:

            aggregated_state_dict[key] = (
                reference_tensor.clone()
            )

    return (
        aggregated_state_dict,
        fedavg_weights,
        total_samples
    )


# =====================================================================
# FEDPROX INFORMATION
# =====================================================================

def explain_fedprox():
    """
    Print the conceptual FedProx formulation.

    NOTE:
        FedProx is NOT a different server aggregation formula.

        It changes the CLIENT'S local optimization objective.

    Standard local objective:

        L_local(w)

    FedProx objective:

        L_local(w)
        + (mu / 2) * ||w - w_global||^2
    """

    print("\n" + "=" * 70)
    print("FEDPROX FORMULATION")
    print("=" * 70)

    print(
        "\nFedProx uses the same sample-weighted server aggregation "
        "as FedAvg."
    )

    print(
        "\nThe difference is in local client optimization:"
    )

    print(
        "\n  FedAvg:"
    )

    print(
        "      L = L_local"
    )

    print(
        "\n  FedProx:"
    )

    print(
        "      L = L_local + (mu / 2) * ||w - w_global||^2"
    )

    print(
        "\nwhere:"
    )

    print(
        "  w        = current local model parameters"
    )

    print(
        "  w_global = global model parameters received "
        "at round start"
    )

    print(
        "  mu       = FedProx proximal coefficient"
    )

    print(
        "\nFedProx local training will be implemented "
        "in the client-side experiment."
    )


# =====================================================================
# MATHEMATICAL FEDAVG TEST
# =====================================================================

def test_fedavg_math():
    """
    Verify FedAvg using a tiny deterministic example.
    """

    print("\n" + "=" * 70)
    print("TEST 1 - FEDAVG MATHEMATICAL VALIDATION")
    print("=" * 70)

    # ---------------------------------------------------------------
    # Client 0
    #
    # parameter = 1
    # samples   = 25
    #
    # Client 1
    #
    # parameter = 3
    # samples   = 75
    #
    # Expected:
    #
    # 0.25 * 1 + 0.75 * 3 = 2.5
    # ---------------------------------------------------------------

    client_results = [

        {
            "client_id": 0,

            "num_samples": 25,

            "state_dict": {
                "weight": torch.tensor(
                    [1.0]
                )
            }
        },

        {
            "client_id": 1,

            "num_samples": 75,

            "state_dict": {
                "weight": torch.tensor(
                    [3.0]
                )
            }
        }
    ]

    aggregated, weights, total = (
        fedavg_aggregate(
            client_results
        )
    )

    expected = 2.5

    actual = (
        aggregated["weight"].item()
    )

    print(
        f"\nClient 0 contribution: "
        f"{weights[0] * 100:.2f}%"
    )

    print(
        f"Client 1 contribution: "
        f"{weights[1] * 100:.2f}%"
    )

    print(
        f"Total samples: {total}"
    )

    print(
        f"Expected value: {expected:.4f}"
    )

    print(
        f"Actual value:   {actual:.4f}"
    )

    assert abs(
        actual - expected
    ) < 1e-6

    assert abs(
        sum(weights.values()) - 1.0
    ) < 1e-8

    print(
        "\n✓ FedAvg mathematical test PASSED"
    )


# =====================================================================
# REAL PARTITION WEIGHT TEST
# =====================================================================

def test_real_partition_weights():
    """
    Verify FedAvg weights using the exact six client sample counts
    from the current alpha=0.3 partition.

    Expected:

        Client 0 = 248
        Client 1 = 734
        Client 2 = 838
        Client 3 = 957
        Client 4 = 73
        Client 5 = 5262

        Total = 8112
    """

    print("\n" + "=" * 70)
    print("TEST 2 - REAL PARTITION FEDAVG WEIGHTS")
    print("=" * 70)

    sample_counts = [
        248,
        734,
        838,
        957,
        73,
        5262,
    ]

    expected_total = 8112

    client_results = []

    for client_id, count in enumerate(
        sample_counts
    ):

        # We only need the metadata for this test.
        client_results.append(
            {
                "client_id": client_id,
                "num_samples": count,
                "state_dict": {
                    "dummy": torch.tensor(
                        [float(client_id)]
                    )
                }
            }
        )

    weights = calculate_fedavg_weights(
        client_results
    )

    total = sum(
        result["num_samples"]
        for result in client_results
    )

    assert total == expected_total, (
        f"Expected {expected_total} samples, "
        f"got {total}."
    )

    print(
        f"\nTotal samples: {total}"
    )

    print(
        "\nFedAvg client contributions:"
    )

    for client_id in range(
        len(sample_counts)
    ):

        count = sample_counts[
            client_id
        ]

        weight = weights[
            client_id
        ]

        print(
            f"  Client {client_id}: "
            f"{count:5d} samples | "
            f"{weight * 100:6.2f}%"
        )

    # ---------------------------------------------------------------
    # Expected approximate values
    # ---------------------------------------------------------------

    expected_weights = [
        248 / 8112,
        734 / 8112,
        838 / 8112,
        957 / 8112,
        73 / 8112,
        5262 / 8112,
    ]

    for client_id in range(6):

        assert abs(
            weights[client_id]
            - expected_weights[client_id]
        ) < 1e-10

    print(
        "\n✓ Real partition weighting test PASSED"
    )


# =====================================================================
# REAL FEDAVG AGGREGATION TEST
# =====================================================================

def test_real_aggregation():
    """
    Test aggregation using six synthetic state dictionaries with
    the REAL client sample counts.

    This verifies that the exact partition imbalance is handled
    correctly by the aggregation implementation.
    """

    print("\n" + "=" * 70)
    print("TEST 3 - REAL-SIZE FEDAVG AGGREGATION")
    print("=" * 70)

    sample_counts = [
        248,
        734,
        838,
        957,
        73,
        5262,
    ]

    client_results = []

    # ---------------------------------------------------------------
    # Each synthetic client has a scalar parameter equal to
    # client_id + 1.
    #
    # FedAvg should therefore produce the weighted average.
    # ---------------------------------------------------------------

    for client_id, count in enumerate(
        sample_counts
    ):

        client_results.append(
            {
                "client_id": client_id,

                "num_samples": count,

                "state_dict": {
                    "weight": torch.tensor(
                        [float(client_id + 1)]
                    )
                }
            }
        )

    aggregated, weights, total = (
        fedavg_aggregate(
            client_results
        )
    )

    # ---------------------------------------------------------------
    # Manually calculate expected answer.
    # ---------------------------------------------------------------

    expected = sum(
        (
            count / 8112
        ) * (client_id + 1)
        for client_id, count
        in enumerate(sample_counts)
    )

    actual = (
        aggregated["weight"].item()
    )

    print(
        f"\nExpected weighted value: "
        f"{expected:.8f}"
    )

    print(
        f"Actual weighted value:   "
        f"{actual:.8f}"
    )

    print(
        f"Total samples: {total}"
    )

    assert abs(
        actual - expected
    ) < 1e-6

    print(
        "\n✓ Real-size FedAvg aggregation "
        "test PASSED"
    )


# =====================================================================
# MODEL STATE DICTIONARY TEST
# =====================================================================

def test_model_state_dictionary():
    """
    Verify that the real HybridClassifier can be constructed and that
    its state_dict can participate in the aggregation pipeline.
    """

    print("\n" + "=" * 70)
    print("TEST 4 - REAL HYBRID MODEL STATE DICTIONARY")
    print("=" * 70)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"\nDevice: {device}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    # ---------------------------------------------------------------
    # Build model
    # ---------------------------------------------------------------

    model = HybridClassifier(
        num_classes=config.NUM_CLASSES,
        handcrafted_feature_dim=(
            config.HANDCRAFTED_FEATURE_DIM
        ),
        pretrained=False,
    ).to(device)

    reference_state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }

    # ---------------------------------------------------------------
    # Create two synthetic local copies
    # ---------------------------------------------------------------

    client_0_state = {
        key: value.clone()
        for key, value in reference_state.items()
    }

    client_1_state = {
        key: value.clone()
        for key, value in reference_state.items()
    }

    # ---------------------------------------------------------------
    # Modify one floating-point parameter in each client.
    # ---------------------------------------------------------------

    modified_key = None

    for key, tensor in client_0_state.items():

        if tensor.is_floating_point():

            client_0_state[key] = (
                tensor + 1.0
            )

            client_1_state[key] = (
                tensor + 3.0
            )

            modified_key = key

            break

    if modified_key is None:
        raise RuntimeError(
            "Could not find a floating-point "
            "parameter for model test."
        )

    # ---------------------------------------------------------------
    # Aggregate using equal sample counts.
    #
    # Expected modified parameter:
    #
    # reference + (1 + 3)/2
    # = reference + 2
    # ---------------------------------------------------------------

    client_results = [

        {
            "client_id": 0,
            "num_samples": 100,
            "state_dict": client_0_state
        },

        {
            "client_id": 1,
            "num_samples": 100,
            "state_dict": client_1_state
        }
    ]

    aggregated, weights, total = (
        fedavg_aggregate(
            client_results,
            reference_state_dict=reference_state
        )
    )

    expected_tensor = (
        reference_state[modified_key]
        + 2.0
    )

    actual_tensor = (
        aggregated[modified_key]
    )

    assert torch.allclose(
        actual_tensor,
        expected_tensor,
        atol=1e-6
    )

    print(
        f"\nTested parameter: "
        f"{modified_key}"
    )

    print(
        "Equal-weight model aggregation: PASSED"
    )

    print(
        "State-dict key/shape validation: PASSED"
    )

    del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =====================================================================
# MAIN
# =====================================================================

def main():

    print("=" * 70)
    print(
        "PHASE 2 - STEP 10: "
        "FEDAVG SERVER / AGGREGATION SELF-TEST"
    )
    print("=" * 70)

    # -----------------------------------------------------------------
    # Validate configuration
    # -----------------------------------------------------------------

    validate_config()

    print(
        "\nConfiguration validation: PASSED"
    )

    print(
        f"  Clients: "
        f"{config.NUM_CLIENTS}"
    )

    print(
        f"  Classes: "
        f"{config.NUM_CLASSES}"
    )

    print(
        f"  Handcrafted features: "
        f"{config.HANDCRAFTED_FEATURE_DIM}"
    )

    # -----------------------------------------------------------------
    # Explain FedProx
    # -----------------------------------------------------------------

    explain_fedprox()

    # -----------------------------------------------------------------
    # Test 1
    # -----------------------------------------------------------------

    start_time = time.perf_counter()

    test_fedavg_math()

    # -----------------------------------------------------------------
    # Test 2
    # -----------------------------------------------------------------

    test_real_partition_weights()

    # -----------------------------------------------------------------
    # Test 3
    # -----------------------------------------------------------------

    test_real_aggregation()

    # -----------------------------------------------------------------
    # Test 4
    # -----------------------------------------------------------------

    test_model_state_dictionary()

    # -----------------------------------------------------------------
    # Finish
    # -----------------------------------------------------------------

    elapsed = (
        time.perf_counter()
        - start_time
    )

    print("\n" + "=" * 70)
    print(
        "PHASE 2 - STEP 10 COMPLETE"
    )
    print("=" * 70)

    print(
        "\n✓ FedAvg mathematical aggregation: PASSED"
    )

    print(
        "✓ Real α=0.3 client weighting: PASSED"
    )

    print(
        "✓ Real-size weighted aggregation: PASSED"
    )

    print(
        "✓ HybridClassifier state_dict test: PASSED"
    )

    print(
        "✓ State-dict compatibility checks: PASSED"
    )

    print(
        "\nFedAvg uses exact sample-size weighting."
    )

    print(
        "FedProx will use the same server aggregation "
        "but a different local client objective."
    )

    print(
        "\nDifferential Privacy has NOT been added."
    )

    print(
        "\nTotal Step 10 test time: "
        f"{elapsed:.2f} seconds"
    )

    print(
        "\nNext:"
    )

    print(
        "  11_train_federated.py"
    )

    print(
        "  -> Multi-round FedAvg training"
    )


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    main()