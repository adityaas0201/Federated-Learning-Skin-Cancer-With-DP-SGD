"""
config.py
---------
Central configuration for the Federated Learning skin-cancer project.
"""

import os


# =====================================================================
# DATA PATHS
# =====================================================================

RAW_DATA_DIR = r"C:\Users\arula\OneDrive\Desktop\FL Skin Cancer\HAM10000"

METADATA_FILE = os.path.join(
    RAW_DATA_DIR,
    "HAM10000_metadata"
)

IMAGE_DIRS = [
    os.path.join(RAW_DATA_DIR, "HAM10000_images_part_1"),
    os.path.join(RAW_DATA_DIR, "HAM10000_images_part_2"),
]


# =====================================================================
# OUTPUT
# =====================================================================

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "outputs"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_SAVE_DIR = os.path.join(OUTPUT_DIR, "models")
TRAINING_LOG_DIR = os.path.join(OUTPUT_DIR, "training_logs")
HANDCRAFTED_FEATURE_DIR = os.path.join(
    OUTPUT_DIR, "handcrafted_features"
)

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(TRAINING_LOG_DIR, exist_ok=True)
os.makedirs(HANDCRAFTED_FEATURE_DIR, exist_ok=True)


# =====================================================================
# CLASSES
# =====================================================================

CLASS_NAMES = {
    "nv": "Melanocytic nevi",
    "mel": "Melanoma",
    "bkl": "Benign keratosis-like lesions",
    "bcc": "Basal cell carcinoma",
    "akiec": "Actinic keratoses / Bowen's disease",
    "vasc": "Vascular lesions",
    "df": "Dermatofibroma",
}

CLASS_LIST = list(CLASS_NAMES.keys())
NUM_CLASSES = 7


# =====================================================================
# IMAGE / PREPROCESSING
# =====================================================================

IMAGE_SIZE = 224

NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

TEST_SPLIT = 0.10
VAL_SPLIT = 0.10


# =====================================================================
# HANDCRAFTED FEATURES
# =====================================================================

GLCM_DISTANCES = [1]

GLCM_ANGLES = [
    0,
    0.785398,
    1.570796,
    2.356194
]

GLCM_PROPERTIES = [
    "contrast",
    "dissimilarity",
    "homogeneity",
    "energy",
    "correlation",
    "ASM"
]

LBP_RADIUS = 1
LBP_POINTS = 8 * LBP_RADIUS
LBP_METHOD = "uniform"

HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (32, 32)
HOG_CELLS_PER_BLOCK = (2, 2)

HANDCRAFTED_FEATURE_DIM = 1320


# =====================================================================
# MODEL
# =====================================================================

MOBILENET_EMBEDDING_DIM = 576


# =====================================================================
# CENTRALIZED TRAINING
# =====================================================================

NUM_CENTRALIZED_EPOCHS = 35
EARLY_STOPPING_PATIENCE = 8


# =====================================================================
# FEDERATED LEARNING
# =====================================================================

NUM_CLIENTS = 6
DIRICHLET_ALPHA = 0.3

NUM_FEDERATED_ROUNDS = 30
NUM_LOCAL_EPOCHS = 3



# =====================================================================
# TRAINING
# =====================================================================

BATCH_SIZE = 16

LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-4

LABEL_SMOOTHING = 0.02

USE_AMP = True

LR_SCHEDULER_FACTOR = 0.5
LR_SCHEDULER_PATIENCE = 2
LR_SCHEDULER_MIN_LR = 1e-6

GRADIENT_CLIP_MAX_NORM = 1.0


# =====================================================================
# REPRODUCIBILITY
# =====================================================================

RANDOM_SEED = 42

# Existing 11A-v3 optimization values -- unchanged.
LEARNING_RATE = 5e-5

FED_MIN_LEARNING_RATE = 1e-7
FED_MAX_LEARNING_RATE = 2e-4
FED_LR_ADAPTATION_RATE = 1.0

FED_CLASS_WEIGHT_POWER = 0.5
FED_MIN_CLASS_WEIGHT = 0.25
FED_MAX_CLASS_WEIGHT = 5.0

FED_LABEL_SMOOTHING = 0.02

# New in 11A-v4.
# These affect ONLY the local federated training sampler.
FED_SAMPLER_WEIGHT_POWER = 0.5
FED_MIN_SAMPLER_WEIGHT = 0.25
FED_MAX_SAMPLER_WEIGHT = 5.0

FED_EXPERIMENT_TAG = "11A_v4_aug_sampler_class_balanced"

SERVER_LR = 1e-3
BETA1     = 0.9
BETA2     = 0.99
EPSILON   = 1e-8

# ============================================================
# FEDERATED TRAINING
# ============================================================

NUM_CLIENTS = 6
CLIENTS_PER_ROUND = 6

FEDERATED_ROUNDS = 20
LOCAL_EPOCHS = 5

LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-4

# ============================================================
# CLIENT ADAPTIVE LEARNING RATE
# ============================================================

FED_MIN_LEARNING_RATE = 1e-7
FED_MAX_LEARNING_RATE = 2e-4
FED_LR_ADAPTATION_RATE = 1.0

# ============================================================
# FEDPROX
# ============================================================

# Adaptive FedProx is used.
# No fixed FEDPROX_MU is required by the v9 driver.

# ============================================================
# CLIENT CLASS BALANCING
# ============================================================

FED_CLASS_WEIGHT_POWER = 0.5
FED_MIN_CLASS_WEIGHT = 0.25
FED_MAX_CLASS_WEIGHT = 5.0

FED_LABEL_SMOOTHING = 0.02

# ============================================================
# HARD-CODED FEDADAM SERVER
# ============================================================

FEDADAM_SERVER_LR = 1e-3
FEDADAM_BETA1 = 0.9
FEDADAM_BETA2 = 0.99
FEDADAM_EPSILON = 1e-8

# ============================================================
# EXPERIMENT
# ============================================================

FED_EXPERIMENT_TAG = "11A_v9_fedadam"

# Local training
FED_LOCAL_SAMPLE_MULTIPLIER = 2.0
LOCAL_EPOCHS = 5

# Client LR
LEARNING_RATE = 5e-5
FED_MIN_LEARNING_RATE = 1e-7
FED_MAX_LEARNING_RATE = 2e-4


# FedProx
FEDPROX_MU_MIN = 1e-4
FEDPROX_MU_MAX = 5e-2

# Auto-FedYogi
FEDYOGI_INITIAL_LR = 1e-3
FEDYOGI_MIN_LR = 1e-4
FEDYOGI_MAX_LR = 3e-3

FEDYOGI_BETA1 = 0.9
FEDYOGI_INITIAL_BETA2 = 0.99
FEDYOGI_MIN_BETA2 = 0.95
FEDYOGI_MAX_BETA2 = 0.999
FEDYOGI_EPSILON = 1e-8

# β2 smoothing
FEDYOGI_BETA2_SMOOTHING = 0.2

# Adaptive aggregation
CLIENT_WEIGHT_GAMMA = 0.5
CLIENT_ENTROPY_STRENGTH = 0.5
CLIENT_RELIABILITY_STRENGTH = 0.5

# Prevent extreme aggregation
CLIENT_WEIGHT_MIN = 0.5
CLIENT_WEIGHT_MAX = 2.0



LEARNING_RATE = 1e-4

FED_MIN_LEARNING_RATE = 1e-5
FED_MAX_LEARNING_RATE = 2e-4

FED_LR_ADAPTATION_RATE = 0.25

BACKBONE_LR_RATIO = 0.10

# ============================================================
# DP FEDERATED LEARNING
# ============================================================

DP_NUM_ROUNDS = 15
DP_LOCAL_EPOCHS = 5
DP_BATCH_SIZE = 4

DP_TARGET_EPSILON = 8.0
DP_TARGET_DELTA = 1e-5
DP_MAX_GRAD_NORM = 1.0

DP_HEAD_LR = 1e-4
DP_BACKBONE_LR_RATIO = 0.10

DP_FEDPROX_MU = 0.01

DP_WEIGHT_DECAY = 1e-4

DP_CLASS_WEIGHT_POWER = 0.50
DP_MIN_CLASS_WEIGHT = 0.25
DP_MAX_CLASS_WEIGHT = 5.0

DP_LABEL_SMOOTHING = 0.02

DP_OPACUS_VERSION = "1.6.0"
DP_ACCOUNTANT = "rdp"
DP_SAMPLING = "poisson"
DP_GRAD_SAMPLE_MODE = "ghost"
DP_CLIPPING = "flat"


# Dataset / clients
NUM_CLIENTS = 6
NUM_FEDERATED_CLIENTS = 6

# Classes
NUM_CLASSES = 7
CLASS_LIST = ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"]

# Handcrafted features
HANDCRAFTED_FEATURE_DIM = 1320

# Reproducibility
RANDOM_SEED = 42

# ============================================================
# ALPHA = 1.0 PARTITION
# ============================================================

DIRICHLET_ALPHA = 1.0
FED_DIRICHLET_ALPHA = 1.0
DIRICHLET_PARTITION_ALPHA = 1.0

# ============================================================
# FIXED GLOBAL CLASS-BALANCED LOSS
# ============================================================

FED_CLASS_WEIGHT_POWER = 0.5
FED_MIN_CLASS_WEIGHT = 0.25
FED_MAX_CLASS_WEIGHT = 5.0

FED_LABEL_SMOOTHING = 0.02
LABEL_SMOOTHING = 0.02

# ============================================================
# OPTIMIZER
# ============================================================

FED_WEIGHT_DECAY = 1e-4
WEIGHT_DECAY = 1e-4

# ============================================================
# SWIN TRANSFORMER V2 TINY - CENTRALIZED BASELINE
# ============================================================

SWIN_BATCH_SIZE = 4
SWIN_GRADIENT_ACCUMULATION_STEPS = 4

SWIN_EPOCHS = 50
SWIN_EARLY_STOPPING_PATIENCE = 8

SWIN_LEARNING_RATE = 1e-4
SWIN_WEIGHT_DECAY = 1e-4

SWIN_LABEL_SMOOTHING = 0.05

SWIN_LR_SCHEDULER_FACTOR = 0.5
SWIN_LR_SCHEDULER_PATIENCE = 3
SWIN_LR_SCHEDULER_MIN_LR = 1e-6

SWIN_GRAD_CLIP = 1.0