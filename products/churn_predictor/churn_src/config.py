"""
Central configuration for the Customer Churn Predictor project.
Every other module imports paths / column names / hyperparameters from here
so there is a single source of truth.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_PATH = os.path.join(DATA_DIR, "customer_churn_data.csv")

MODELS_DIR = os.path.join(BASE_DIR, "models")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

MODEL_PATH = os.path.join(MODELS_DIR, "lstm_churn_model.keras")
ONNX_MODEL_PATH = os.path.join(MODELS_DIR, "lstm_churn_model.onnx")
SCALER_PATH = os.path.join(MODELS_DIR, "feature_scaler.pkl")
ENCODERS_PATH = os.path.join(MODELS_DIR, "categorical_encoders.pkl")
FEATURE_LIST_PATH = os.path.join(MODELS_DIR, "feature_columns.json")
SHAP_BACKGROUND_PATH = os.path.join(MODELS_DIR, "shap_background.npy")
METRICS_PATH = os.path.join(ARTIFACTS_DIR, "evaluation_metrics.json")

for d in (DATA_DIR, MODELS_DIR, ARTIFACTS_DIR):
    os.makedirs(d, exist_ok=True)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
ID_COL = "CustomerID"
TIME_COL = "Month"
TARGET_COL = "Churn"

NUMERIC_FEATURES = [
    "LoginCount",
    "SessionDuration",
    "PurchaseAmount",
    "MonthlyCharge",
    "DataUsage",
    "Complaints",
    "SupportCalls",
    "PaymentDelay",
    "Tenure",
]

CATEGORICAL_FEATURES = ["ContractType", "PlanType"]

CONTRACT_TYPES = ["Month-to-Month", "One Year", "Two Year"]
PLAN_TYPES = ["Basic", "Standard", "Premium"]

# Engineered features added by src/feature_engineering.py
ENGINEERED_FEATURES = [
    "LoginCount_delta",
    "SessionDuration_delta",
    "DataUsage_delta",
    "PurchaseAmount_delta",
    "Complaints_roll3",
    "SupportCalls_roll3",
    "PaymentDelay_roll3",
    "AvgSessionPerLogin",
    "ChargePerGB",
    "EngagementScore",
]

# Final ordered feature list used as LSTM input (numeric + engineered + encoded categoricals)
def get_full_feature_list():
    encoded_cats = [f"{c}_encoded" for c in CATEGORICAL_FEATURES]
    return NUMERIC_FEATURES + ENGINEERED_FEATURES + encoded_cats

# ---------------------------------------------------------------------------
# Sequence / model hyperparameters
# ---------------------------------------------------------------------------
MAX_SEQ_LEN = 12          # months of history fed to the LSTM
RANDOM_STATE = 42

LSTM_UNITS_1 = 64
LSTM_UNITS_2 = 32
DROPOUT_RATE = 0.3
DENSE_UNITS = 16
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
EPOCHS = 50
EARLY_STOPPING_PATIENCE = 8

# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------
NUM_CUSTOMERS = 3000
MIN_MONTHS = 3
MAX_MONTHS = 12
BASE_CHURN_RATE = 0.27
