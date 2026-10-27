"""
06_dataset.py
-------------
PHASE 2 - STEP 6: DATASET AND DATALOADER

Hybrid input:
    Image + 1320 handcrafted features + label

Training:
    Image augmentation is applied ONLY to training images.

Validation/Test:
    No augmentation.

Important:
    StandardScaler is fitted ONLY on TRAIN handcrafted features.
"""

import os
import sys
import time
import pickle

import cv2
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler

from torchvision import transforms


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
# PATHS
# ================================================================

FEATURE_DIR = config.HANDCRAFTED_FEATURE_DIR

SCALER_PATH = os.path.join(
    FEATURE_DIR,
    "feature_scaler.pkl"
)

SCALED_FEATURE_DIR = os.path.join(
    FEATURE_DIR,
    "scaled"
)

os.makedirs(
    SCALED_FEATURE_DIR,
    exist_ok=True
)


# ================================================================
# FEATURE DIMENSION
# ================================================================

EXPECTED_FEATURE_DIM = config.HANDCRAFTED_FEATURE_DIM


# ================================================================
# DEVICE
# ================================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ================================================================
# IMAGE AUGMENTATION
# ================================================================
#
# Dermoscopic images can naturally appear with:
#   - different orientations
#   - small rotations
#   - different crops/scales
#   - mild lighting/contrast differences
#
# We therefore use moderate augmentation.
#
# IMPORTANT:
# Handcrafted features were extracted from the original image.
# Therefore we avoid aggressive color transformations.
#
# Augmentation is applied ONLY during training.
# ================================================================

TRAIN_TRANSFORM = transforms.Compose([

    # Orientation augmentation
    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomVerticalFlip(
        p=0.5
    ),

    # Rotation augmentation
    transforms.RandomRotation(
        degrees=20
    ),

    # Mild spatial variation
    transforms.RandomResizedCrop(
        size=config.IMAGE_SIZE,
        scale=(0.90, 1.0),
        ratio=(0.95, 1.05)
    ),

])


# No augmentation for validation/test.
EVAL_TRANSFORM = transforms.Compose([])


# ================================================================
# LOAD FEATURE CACHE
# ================================================================

def load_feature_cache(npz_path):
    """
    Loads raw handcrafted features.

    Expected:
        image_ids
        features
    """

    if not os.path.exists(npz_path):

        raise FileNotFoundError(
            f"\nFeature cache not found:\n"
            f"{npz_path}\n\n"
            f"Run 05_handcrafted_features.py first."
        )

    data = np.load(
        npz_path,
        allow_pickle=False
    )

    image_ids = data["image_ids"]
    features = data["features"]

    if len(image_ids) != len(features):

        raise ValueError(
            f"Feature cache mismatch:\n"
            f"image_ids = {len(image_ids)}\n"
            f"features  = {len(features)}"
        )

    if features.ndim != 2:

        raise ValueError(
            f"Expected 2-D feature matrix, "
            f"got {features.shape}"
        )

    if features.shape[1] != EXPECTED_FEATURE_DIM:

        raise ValueError(
            f"Unexpected feature dimension.\n"
            f"Expected: {EXPECTED_FEATURE_DIM}\n"
            f"Found:    {features.shape[1]}"
        )

    if not np.isfinite(features).all():

        raise ValueError(
            f"NaN or Inf detected in:\n{npz_path}"
        )

    return {
        str(image_id): features[i].astype(
            np.float32
        )
        for i, image_id in enumerate(image_ids)
    }


# ================================================================
# LOAD SCALER
# ================================================================

def load_scaler():

    if not os.path.exists(SCALER_PATH):

        raise FileNotFoundError(
            f"\nScaler not found:\n"
            f"{SCALER_PATH}\n\n"
            f"Run Step 6 once to create it."
        )

    with open(
        SCALER_PATH,
        "rb"
    ) as f:

        return pickle.load(f)


# ================================================================
# FIT TRAIN SCALER
# ================================================================

def fit_train_scaler(train_lookup):

    print(
        "\nFitting StandardScaler on TRAIN "
        "features only..."
    )

    train_features = np.stack(
        list(train_lookup.values())
    ).astype(np.float32)

    scaler = StandardScaler()

    scaler.fit(
        train_features
    )

    with open(
        SCALER_PATH,
        "wb"
    ) as f:

        pickle.dump(
            scaler,
            f
        )

    print(
        f"Scaler fitted on "
        f"{len(train_features)} samples."
    )

    print(
        f"Scaler saved to:\n"
        f"  {SCALER_PATH}"
    )

    return scaler


# ================================================================
# SCALE AND CACHE FEATURES
# ================================================================

def scale_and_cache_features(
    split_name,
    feature_lookup,
    scaler
):

    output_path = os.path.join(
        SCALED_FEATURE_DIR,
        f"{split_name}_features_scaled.npz"
    )

    image_ids = list(
        feature_lookup.keys()
    )

    raw_features = np.stack([
        feature_lookup[x]
        for x in image_ids
    ]).astype(np.float32)

    scaled_features = scaler.transform(
        raw_features
    ).astype(np.float32)

    if not np.isfinite(
        scaled_features
    ).all():

        raise ValueError(
            f"NaN/Inf produced while scaling "
            f"{split_name}."
        )

    np.savez_compressed(
        output_path,
        image_ids=np.array(
            image_ids
        ),
        features=scaled_features
    )

    print(
        f"\n{split_name} scaled features:"
        f" {scaled_features.shape}"
    )

    return {
        str(image_ids[i]): scaled_features[i]
        for i in range(len(image_ids))
    }


# ================================================================
# LOAD OR CREATE SCALED FEATURES
# ================================================================

def prepare_scaled_features(
    split_name,
    raw_feature_path,
    scaler
):

    scaled_path = os.path.join(
        SCALED_FEATURE_DIR,
        f"{split_name}_features_scaled.npz"
    )

    if os.path.exists(scaled_path):

        print(
            f"\nUsing existing scaled feature cache:\n"
            f"  {scaled_path}"
        )

        data = np.load(
            scaled_path,
            allow_pickle=False
        )

        image_ids = data["image_ids"]
        features = data["features"]

        if features.ndim != 2:

            raise ValueError(
                f"Invalid scaled feature shape:"
                f" {features.shape}"
            )

        if features.shape[1] != EXPECTED_FEATURE_DIM:

            raise ValueError(
                f"Invalid feature dimension:"
                f" {features.shape}"
            )

        if not np.isfinite(
            features
        ).all():

            raise ValueError(
                f"NaN/Inf found in:"
                f" {scaled_path}"
            )

        return {
            str(image_ids[i]): features[i].astype(
                np.float32
            )
            for i in range(len(image_ids))
        }

    raw_lookup = load_feature_cache(
        raw_feature_path
    )

    return scale_and_cache_features(
        split_name,
        raw_lookup,
        scaler
    )


# ================================================================
# DATASET
# ================================================================

class SkinLesionDataset(Dataset):

    def __init__(
        self,
        dataframe,
        feature_lookup,
        transform=None
    ):

        self.df = dataframe.reset_index(
            drop=True
        )

        self.feature_lookup = feature_lookup

        self.transform = transform

        self.class_to_idx = {
            class_name: index
            for index, class_name
            in enumerate(config.CLASS_LIST)
        }

        # --------------------------------------------------------
        # Verify handcrafted features
        # --------------------------------------------------------

        missing_features = [
            image_id
            for image_id in self.df["image_id"]
            if str(image_id) not in self.feature_lookup
        ]

        if missing_features:

            raise ValueError(
                f"{len(missing_features)} images are "
                f"missing handcrafted features.\n"
                f"Examples: {missing_features[:5]}"
            )

        # --------------------------------------------------------
        # Verify labels
        # --------------------------------------------------------

        unknown_labels = (
            set(self.df["label"])
            -
            set(self.class_to_idx.keys())
        )

        if unknown_labels:

            raise ValueError(
                f"Unknown labels found:"
                f" {unknown_labels}"
            )

    # ============================================================
    # LENGTH
    # ============================================================

    def __len__(self):

        return len(
            self.df
        )

    # ============================================================
    # LOAD IMAGE
    # ============================================================

    @staticmethod
    def load_image(filepath):

        image = cv2.imread(
            filepath
        )

        if image is None:

            raise ValueError(
                f"Could not read image:\n"
                f"{filepath}"
            )

        # --------------------------------------------------------
        # BGR -> RGB
        # --------------------------------------------------------

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        # --------------------------------------------------------
        # uint8 [0,255] -> float [0,1]
        # --------------------------------------------------------

        image = image.astype(
            np.float32
        ) / 255.0

        # --------------------------------------------------------
        # Safety resize
        # --------------------------------------------------------

        if (
            image.shape[0] != config.IMAGE_SIZE
            or
            image.shape[1] != config.IMAGE_SIZE
        ):

            image = cv2.resize(
                image,
                (
                    config.IMAGE_SIZE,
                    config.IMAGE_SIZE
                ),
                interpolation=cv2.INTER_AREA
            )

        # --------------------------------------------------------
        # HWC -> CHW
        # --------------------------------------------------------

        image = np.transpose(
            image,
            (2, 0, 1)
        )

        return torch.from_numpy(
            image.copy()
        ).float()

    # ============================================================
    # GET SAMPLE
    # ============================================================

    def __getitem__(
        self,
        index
    ):

        row = self.df.iloc[
            index
        ]

        # --------------------------------------------------------
        # IMAGE
        # --------------------------------------------------------

        image = self.load_image(
            row["filepath"]
        )

        # --------------------------------------------------------
        # AUGMENTATION
        # --------------------------------------------------------
        #
        # Applied only when training=True.
        #
        # The image is still in [0,1] range here.
        # --------------------------------------------------------

        if self.transform is not None:

            image = self.transform(
                image
            )

        # --------------------------------------------------------
        # IMAGENET NORMALIZATION
        # --------------------------------------------------------

        mean = torch.tensor(
            config.NORMALIZE_MEAN,
            dtype=torch.float32
        ).view(
            3,
            1,
            1
        )

        std = torch.tensor(
            config.NORMALIZE_STD,
            dtype=torch.float32
        ).view(
            3,
            1,
            1
        )

        image = (
            image - mean
        ) / std

        # --------------------------------------------------------
        # HANDCRAFTED FEATURES
        # --------------------------------------------------------
        #
        # Already standardized using the train-fitted scaler.
        # --------------------------------------------------------

        features = self.feature_lookup[
            str(row["image_id"])
        ]

        features = torch.from_numpy(
            features.copy()
        ).float()

        # --------------------------------------------------------
        # LABEL
        # --------------------------------------------------------

        label = self.class_to_idx[
            row["label"]
        ]

        label = torch.tensor(
            label,
            dtype=torch.long
        )

        return (
            image,
            features,
            label
        )


# ================================================================
# LOAD SPLIT CSV
# ================================================================

def load_split_csv(
    split_name
):

    csv_path = os.path.join(
        config.OUTPUT_DIR,
        f"{split_name}_split.csv"
    )

    if not os.path.exists(
        csv_path
    ):

        raise FileNotFoundError(
            f"\nCould not find:\n"
            f"{csv_path}"
        )

    return pd.read_csv(
        csv_path
    )


# ================================================================
# CREATE DATALOADER
# ================================================================

def create_loader(
    dataframe,
    feature_lookup,
    shuffle,
    batch_size=None,
    training=False
):

    if batch_size is None:

        batch_size = config.BATCH_SIZE

    # ------------------------------------------------------------
    # Select transform
    # ------------------------------------------------------------

    if training:

        transform = TRAIN_TRANSFORM

    else:

        transform = EVAL_TRANSFORM

    # ------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------

    dataset = SkinLesionDataset(
        dataframe=dataframe,
        feature_lookup=feature_lookup,
        transform=transform
    )

    # ------------------------------------------------------------
    # DataLoader
    # ------------------------------------------------------------

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=torch.cuda.is_available(),
        num_workers=0,
        drop_last=False
    )

    return loader


# ================================================================
# PREPARE ALL FEATURES
# ================================================================

def prepare_all_features():

    print(
        "\n" + "=" * 60
    )

    print(
        "PREPARING SCALED HANDCRAFTED FEATURES"
    )

    print(
        "=" * 60
    )

    # ------------------------------------------------------------
    # TRAIN
    # ------------------------------------------------------------

    train_feature_path = os.path.join(
        FEATURE_DIR,
        "train_features.npz"
    )

    train_lookup = load_feature_cache(
        train_feature_path
    )

    print(
        f"\nTrain handcrafted features:"
        f" {len(train_lookup)}"
    )

    # IMPORTANT:
    # Fit scaler ONLY on training data.

    scaler = fit_train_scaler(
        train_lookup
    )

    # ------------------------------------------------------------
    # TRAIN scaled features
    # ------------------------------------------------------------

    train_scaled = scale_and_cache_features(
        "train",
        train_lookup,
        scaler
    )

    # ------------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------------

    val_lookup = load_feature_cache(
        os.path.join(
            FEATURE_DIR,
            "val_features.npz"
        )
    )

    val_scaled = scale_and_cache_features(
        "val",
        val_lookup,
        scaler
    )

    # ------------------------------------------------------------
    # TEST
    # ------------------------------------------------------------

    test_lookup = load_feature_cache(
        os.path.join(
            FEATURE_DIR,
            "test_features.npz"
        )
    )

    test_scaled = scale_and_cache_features(
        "test",
        test_lookup,
        scaler
    )

    print(
        "\nScaled feature preparation complete ✓"
    )

    return (
        train_scaled,
        val_scaled,
        test_scaled
    )


# ================================================================
# CENTRALIZED TRAIN LOADER
# ================================================================

def build_centralized_train_loader(
    train_features=None
):

    if train_features is None:

        scaler = load_scaler()

        train_features = prepare_scaled_features(
            "train",
            os.path.join(
                FEATURE_DIR,
                "train_features.npz"
            ),
            scaler
        )

    train_df = load_split_csv(
        "train"
    )

    return create_loader(
        dataframe=train_df,
        feature_lookup=train_features,
        shuffle=True,
        training=True
    )


# ================================================================
# FEDERATED CLIENT LOADER
# ================================================================

def build_client_loader(
    client_id,
    train_features=None
):

    client_csv = os.path.join(
        config.OUTPUT_DIR,
        "client_partitions",
        f"client_{client_id}.csv"
    )

    if not os.path.exists(
        client_csv
    ):

        raise FileNotFoundError(
            f"\nClient CSV not found:\n"
            f"{client_csv}\n\n"
            f"Run 02_dirichlet_partition.py first."
        )

    client_df = pd.read_csv(
        client_csv
    )

    if train_features is None:

        scaler = load_scaler()

        train_features = prepare_scaled_features(
            "train",
            os.path.join(
                FEATURE_DIR,
                "train_features.npz"
            ),
            scaler
        )

    return create_loader(
        dataframe=client_df,
        feature_lookup=train_features,
        shuffle=True,
        training=True
    )


# ================================================================
# EVALUATION LOADER
# ================================================================

def build_eval_loader(
    split_name,
    scaled_features=None
):

    if split_name not in [
        "val",
        "test"
    ]:

        raise ValueError(
            "split_name must be 'val' or 'test'"
        )

    if scaled_features is None:

        scaler = load_scaler()

        scaled_features = prepare_scaled_features(
            split_name,
            os.path.join(
                FEATURE_DIR,
                f"{split_name}_features.npz"
            ),
            scaler
        )

    df = load_split_csv(
        split_name
    )

    return create_loader(
        dataframe=df,
        feature_lookup=scaled_features,
        shuffle=False,
        training=False
    )


# ================================================================
# SANITY CHECK
# ================================================================

def sanity_check():

    print(
        "\n" + "=" * 60
    )

    print(
        "PHASE 2 - STEP 6: DATASET / DATALOADER CHECK"
    )

    print(
        "=" * 60
    )

    print(
        f"\nDevice: {DEVICE}"
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

    # ------------------------------------------------------------
    # Prepare features
    # ------------------------------------------------------------

    start = time.perf_counter()

    (
        train_features,
        val_features,
        test_features
    ) = prepare_all_features()

    print(
        f"\nFeature preparation time:"
        f" {time.perf_counter() - start:.2f}s"
    )

    # ------------------------------------------------------------
    # Load CSVs
    # ------------------------------------------------------------

    train_df = load_split_csv(
        "train"
    )

    val_df = load_split_csv(
        "val"
    )

    test_df = load_split_csv(
        "test"
    )

    print(
        "\nDataset sizes:"
    )

    print(
        f"  Train: {len(train_df)}"
    )

    print(
        f"  Val  : {len(val_df)}"
    )

    print(
        f"  Test : {len(test_df)}"
    )

    # ------------------------------------------------------------
    # Training loader
    # ------------------------------------------------------------

    train_loader = create_loader(
        dataframe=train_df,
        feature_lookup=train_features,
        shuffle=True,
        training=True
    )

    # ------------------------------------------------------------
    # Validation loader
    # ------------------------------------------------------------

    val_loader = create_loader(
        dataframe=val_df,
        feature_lookup=val_features,
        shuffle=False,
        training=False
    )

    # ------------------------------------------------------------
    # Test loader
    # ------------------------------------------------------------

    test_loader = create_loader(
        dataframe=test_df,
        feature_lookup=test_features,
        shuffle=False,
        training=False
    )

    # ------------------------------------------------------------
    # Check train batch
    # ------------------------------------------------------------

    images, features, labels = next(
        iter(train_loader)
    )

    print(
        "\nTrain batch:"
    )

    print(
        f"  Images   : {images.shape}"
    )

    print(
        f"  Features : {features.shape}"
    )

    print(
        f"  Labels   : {labels.shape}"
    )

    expected_image_shape = (
        config.BATCH_SIZE,
        3,
        config.IMAGE_SIZE,
        config.IMAGE_SIZE
    )

    expected_feature_shape = (
        config.BATCH_SIZE,
        EXPECTED_FEATURE_DIM
    )

    if images.shape != expected_image_shape:

        raise ValueError(
            f"Unexpected image shape.\n"
            f"Expected: {expected_image_shape}\n"
            f"Got: {images.shape}"
        )

    if features.shape != expected_feature_shape:

        raise ValueError(
            f"Unexpected feature shape.\n"
            f"Expected: {expected_feature_shape}\n"
            f"Got: {features.shape}"
        )

    if labels.ndim != 1:

        raise ValueError(
            f"Unexpected label shape:"
            f" {labels.shape}"
        )

    # ------------------------------------------------------------
    # Feature statistics
    # ------------------------------------------------------------

    train_feature_matrix = np.stack(
        list(train_features.values())
    )

    mean_value = float(
        train_feature_matrix.mean()
    )

    std_value = float(
        train_feature_matrix.std()
    )

    print(
        "\nScaled TRAIN feature statistics:"
    )

    print(
        f"  Mean: {mean_value:.6f}"
    )

    print(
        f"  Std : {std_value:.6f}"
    )

    if abs(mean_value) > 0.01:

        print(
            "[WARNING] Train feature mean "
            "is not close to zero."
        )

    if abs(std_value - 1.0) > 0.02:

        print(
            "[WARNING] Train feature std "
            "is not close to one."
        )

    # ------------------------------------------------------------
    # Label mapping
    # ------------------------------------------------------------

    print(
        "\nLabel mapping:"
    )

    for index, class_name in enumerate(
        config.CLASS_LIST
    ):

        print(
            f"  {index} -> {class_name}"
        )

    if labels.min() < 0:

        raise ValueError(
            "Negative label detected."
        )

    if labels.max() >= config.NUM_CLASSES:

        raise ValueError(
            "Label exceeds NUM_CLASSES."
        )

    # ------------------------------------------------------------
    # Client 0
    # ------------------------------------------------------------

    client_loader = build_client_loader(
        0,
        train_features
    )

    client_images, client_features, client_labels = next(
        iter(client_loader)
    )

    print(
        "\nClient 0 batch:"
    )

    print(
        f"  Images   : {client_images.shape}"
    )

    print(
        f"  Features : {client_features.shape}"
    )

    print(
        f"  Labels   : {client_labels.shape}"
    )

    # ------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------

    val_batch = next(
        iter(val_loader)
    )

    print(
        "\nValidation batch:"
    )

    print(
        f"  Images   : {val_batch[0].shape}"
    )

    print(
        f"  Features : {val_batch[1].shape}"
    )

    print(
        f"  Labels   : {val_batch[2].shape}"
    )

    # ------------------------------------------------------------
    # Test
    # ------------------------------------------------------------

    test_batch = next(
        iter(test_loader)
    )

    print(
        "\nTest batch:"
    )

    print(
        f"  Images   : {test_batch[0].shape}"
    )

    print(
        f"  Features : {test_batch[1].shape}"
    )

    print(
        f"  Labels   : {test_batch[2].shape}"
    )

    # ------------------------------------------------------------
    # Final
    # ------------------------------------------------------------

    print(
        "\n" + "=" * 60
    )

    print(
        "PHASE 2 - STEP 6 COMPLETE ✓"
    )

    print(
        "=" * 60
    )

    print(
        "\nVerified:"
    )

    print(
        "  ✓ Train-only StandardScaler"
    )

    print(
        "  ✓ Scaled feature caching"
    )

    print(
        "  ✓ RGB image loading"
    )

    print(
        "  ✓ Training augmentation"
    )

    print(
        "  ✓ No augmentation for validation/test"
    )

    print(
        "  ✓ ImageNet normalization"
    )

    print(
        "  ✓ 224x224 image tensors"
    )

    print(
        "  ✓ 1320-D handcrafted features"
    )

    print(
        "  ✓ 7-class label mapping"
    )

    print(
        "  ✓ Centralized DataLoader"
    )

    print(
        "  ✓ Federated client DataLoader"
    )

    print(
        "  ✓ Validation DataLoader"
    )

    print(
        "  ✓ Test DataLoader"
    )

    print(
        "\nNext:"
    )

    print(
        "  python src\\07_model.py"
    )


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":

    sanity_check()