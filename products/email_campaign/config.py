"""
Central configuration for the AI Email Campaign Predictor.
All paths, hyperparameters, and column definitions live here so every
script in the pipeline stays in sync.
"""

import os
import torch

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_CSV_PATH = os.path.join(DATA_DIR, "email_campaigns.csv")
IMAGES_DIR = os.path.join(DATA_DIR, "images")

PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
TRAIN_CSV = os.path.join(PROCESSED_DIR, "train.csv")
VAL_CSV = os.path.join(PROCESSED_DIR, "val.csv")
TEST_CSV = os.path.join(PROCESSED_DIR, "test.csv")

MODELS_DIR = os.path.join(BASE_DIR, "models")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_fusion_model.pt")
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "preprocessor.joblib")
SHAP_BACKGROUND_PATH = os.path.join(MODELS_DIR, "shap_background.joblib")

# ----------------------------------------------------------------------
# Hugging Face Hub fallback for large model artifacts
# ----------------------------------------------------------------------
# best_fusion_model.pt (270MB) is no longer relied on via Git LFS -- GitHub's
# free LFS bandwidth quota (1GB/month) gets exhausted after just ~3-4 deploys
# of a file this size, silently leaving an unresolved LFS pointer on disk
# (which is what caused "invalid load key, 'v'" -- torch.load was trying to
# unpickle the pointer's "version https://..." text instead of a real
# checkpoint). Instead, these files are hosted on Hugging Face Hub and
# downloaded once at startup if missing locally.
#
# Set these via environment variables on Render (Settings -> Environment):
#   HF_MODEL_REPO_ID   e.g. "your-username/core-growth-email-campaign"
#   HF_TOKEN            only needed if the HF repo is private
HF_MODEL_REPO_ID = os.getenv("HF_MODEL_REPO_ID", "")
HF_TOKEN = os.getenv("HF_TOKEN")


def ensure_model_artifacts():
    """Downloads best_fusion_model.pt and preprocessor.joblib from Hugging
    Face Hub into MODELS_DIR if they aren't already present locally.
    No-ops if HF_MODEL_REPO_ID isn't set (e.g. local dev with files already
    in place) or if both files already exist."""
    need_model = not os.path.exists(BEST_MODEL_PATH)
    need_preprocessor = not os.path.exists(PREPROCESSOR_PATH)
    if not (need_model or need_preprocessor):
        return

    if not HF_MODEL_REPO_ID:
        # Nothing we can do -- let the caller's existing file-not-found /
        # torch.load error surface as before.
        return

    from huggingface_hub import hf_hub_download

    os.makedirs(MODELS_DIR, exist_ok=True)

    if need_model:
        print(f"[email_campaign] Downloading best_fusion_model.pt from {HF_MODEL_REPO_ID} ...")
        downloaded = hf_hub_download(
            repo_id=HF_MODEL_REPO_ID,
            filename="best_fusion_model.pt",
            local_dir=MODELS_DIR,
            token=HF_TOKEN,
        )
        if os.path.abspath(downloaded) != os.path.abspath(BEST_MODEL_PATH):
            import shutil
            shutil.move(downloaded, BEST_MODEL_PATH)
        print("[email_campaign] best_fusion_model.pt downloaded.")

    if need_preprocessor:
        print(f"[email_campaign] Downloading preprocessor.joblib from {HF_MODEL_REPO_ID} ...")
        downloaded = hf_hub_download(
            repo_id=HF_MODEL_REPO_ID,
            filename="preprocessor.joblib",
            local_dir=MODELS_DIR,
            token=HF_TOKEN,
        )
        if os.path.abspath(downloaded) != os.path.abspath(PREPROCESSOR_PATH):
            import shutil
            shutil.move(downloaded, PREPROCESSOR_PATH)
        print("[email_campaign] preprocessor.joblib downloaded.")

OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

for d in [DATA_DIR, IMAGES_DIR, PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ----------------------------------------------------------------------
# Dataset generation
# ----------------------------------------------------------------------
N_SYNTHETIC_ROWS = 6000
RANDOM_SEED = 42

SEGMENTS = ["New Subscribers", "Loyal Customers", "Cart Abandoners",
            "High Value", "Dormant Users", "Newsletter Only"]
CAMPAIGN_TYPES = ["Promotional", "Newsletter", "Product Launch",
                   "Re-engagement", "Transactional", "Seasonal Sale"]

# ----------------------------------------------------------------------
# Feature groups
# ----------------------------------------------------------------------
NUMERIC_FEATURES = [
    "audience_size", "send_day", "send_hour", "discount_percent",
    "body_length", "num_links", "num_ctas", "has_personalization",
    "past_open_rate", "past_ctr", "past_conversion_rate",
]

CATEGORICAL_FEATURES = ["segment", "campaign_type"]

TEXT_COLUMNS = ["subject", "body"]
IMAGE_COLUMN = "image_path"

TARGET_COLUMNS = [
    "open_rate", "ctr", "conversion_rate",
    "roi", "unsubscribe_rate", "performance_score",
]

ID_COLUMN = "campaign_id"

# ----------------------------------------------------------------------
# Train / val / test split
# ----------------------------------------------------------------------
TEST_SIZE = 0.15
VAL_SIZE = 0.15  # of the remaining train set

# ----------------------------------------------------------------------
# Model hyperparameters
# ----------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tabular MLP
TABULAR_HIDDEN_DIMS = [128, 64]
TABULAR_OUT_DIM = 64

# Text model (DistilBERT)
TEXT_MODEL_NAME = "distilbert-base-uncased"
TEXT_MAX_LENGTH = 128
TEXT_OUT_DIM = 64
FREEZE_TEXT_BACKBONE = True  # freeze DistilBERT weights, train only projection head

# Image model (EfficientNet-B0)
IMAGE_MODEL_NAME = "efficientnet_b0"
IMAGE_SIZE = 224
IMAGE_OUT_DIM = 64
FREEZE_IMAGE_BACKBONE = True

# Fusion
FUSION_HIDDEN_DIM = 128
FUSION_DROPOUT = 0.3

# Training
BATCH_SIZE = 32
NUM_EPOCHS = 25
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
EARLY_STOPPING_PATIENCE = 5
NUM_WORKERS = 2

# Multi-output loss weights (order must match TARGET_COLUMNS)
TARGET_LOSS_WEIGHTS = {
    "open_rate": 1.0,
    "ctr": 1.0,
    "conversion_rate": 1.0,
    "roi": 0.5,          # roi has larger scale/variance -> down-weighted
    "unsubscribe_rate": 1.0,
    "performance_score": 1.0,
}
