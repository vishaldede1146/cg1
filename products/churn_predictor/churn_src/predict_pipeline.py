"""
Prediction pipeline used by both the FastAPI service and the Streamlit app.

Expected input: a customer's monthly history as a list of month-records
(dicts), each with the raw schema fields (minus CustomerID/Churn, which are
not needed at inference time). At least 1 month is required; up to
MAX_SEQ_LEN most recent months are used, fewer are zero-padded automatically
(the model was trained to handle this).

Example input (JSON):
[
  {"Month": 1, "LoginCount": 20, "SessionDuration": 40, "PurchaseAmount": 60,
   "MonthlyCharge": 55, "DataUsage": 20, "Complaints": 0, "SupportCalls": 0,
   "PaymentDelay": 0, "ContractType": "Month-to-Month", "Tenure": 5, "PlanType": "Standard"},
  ...
]
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import onnxruntime as ort

from churn_src import config
from churn_src.data_cleaning import clean_data
from churn_src.feature_engineering import engineer_features
from churn_src.sequence_preparation import build_sequences
from churn_src.scaling import transform_sequences
from churn_src.explainability import explain_instance, load_background

REQUIRED_FIELDS = [
    "Month", "LoginCount", "SessionDuration", "PurchaseAmount", "MonthlyCharge",
    "DataUsage", "Complaints", "SupportCalls", "PaymentDelay", "ContractType",
    "Tenure", "PlanType",
]


class ValidationError(Exception):
    pass


def validate_input(months: list):
    if not isinstance(months, list) or len(months) == 0:
        raise ValidationError("Input must be a non-empty list of monthly records.")

    for i, rec in enumerate(months):
        missing = [f for f in REQUIRED_FIELDS if f not in rec]
        if missing:
            raise ValidationError(f"Record {i} is missing fields: {missing}")

        if rec["ContractType"] not in config.CONTRACT_TYPES:
            raise ValidationError(
                f"Record {i}: ContractType must be one of {config.CONTRACT_TYPES}"
            )
        if rec["PlanType"] not in config.PLAN_TYPES:
            raise ValidationError(f"Record {i}: PlanType must be one of {config.PLAN_TYPES}")

        for f in config.NUMERIC_FEATURES + ["Month"]:
            try:
                val = float(rec[f])
            except (TypeError, ValueError):
                raise ValidationError(f"Record {i}: field '{f}' must be numeric.")
            if val < 0:
                raise ValidationError(f"Record {i}: field '{f}' cannot be negative.")

    return True


class ChurnPredictor:
    """Loads all artifacts once; call .predict(months) repeatedly for inference."""

    def __init__(self):
        self.session = ort.InferenceSession(config.ONNX_MODEL_PATH, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.scaler = joblib.load(config.SCALER_PATH)
        self.encoders = joblib.load(config.ENCODERS_PATH)
        with open(config.FEATURE_LIST_PATH) as f:
            self.feature_cols = json.load(f)
        self.background = load_background()

    def _prepare_sequence(self, months: list) -> np.ndarray:
        df = pd.DataFrame(months)
        df[config.ID_COL] = "TEMP_CUSTOMER"
        df[config.TARGET_COL] = 0  # placeholder, unused for inference

        df = clean_data(df)
        df_fe, _ = engineer_features(df, encoders=self.encoders)

        X, _, _ = build_sequences(df_fe, self.feature_cols)
        X_scaled = transform_sequences(X, self.scaler)
        return X_scaled

    def predict(self, months: list, explain: bool = True, threshold: float = 0.5) -> dict:
        validate_input(months)
        X_scaled = self._prepare_sequence(months).astype(np.float32)

        prob = float(self.session.run(None, {self.input_name: X_scaled})[0].ravel()[0])
        pred_label = int(prob >= threshold)

        result = {
            "churn_probability": round(prob, 4),
            "churn_prediction": pred_label,
            "risk_level": _risk_level(prob),
        }

        if explain:
            explanation = explain_instance(self.session, self.input_name, self.background,
                                            X_scaled, self.feature_cols)
            result["top_factors"] = [
                {"feature": f, "impact": round(v, 4), "direction": "increases risk" if v > 0 else "decreases risk"}
                for f, v in explanation["top_features"]
            ]
            result["business_insights"] = generate_business_insights(months[-1], explanation, prob)

        return result


def _risk_level(prob: float) -> str:
    if prob >= 0.7:
        return "High"
    if prob >= 0.4:
        return "Medium"
    return "Low"


def generate_business_insights(latest_month: dict, explanation: dict, prob: float) -> list:
    """Rule-based, human-readable recommendations derived from SHAP top factors + raw values."""
    insights = []
    top_feats = dict(explanation["top_features"])

    if prob >= 0.7:
        insights.append("Customer is at HIGH risk of churn — prioritize proactive retention outreach.")
    elif prob >= 0.4:
        insights.append("Customer shows moderate churn risk — monitor closely over the next billing cycle.")
    else:
        insights.append("Customer appears healthy with low churn risk.")

    if top_feats.get("PaymentDelay", 0) > 0 or latest_month.get("PaymentDelay", 0) > 3:
        insights.append("Recent payment delays are a key risk driver — consider flexible billing or a payment reminder call.")

    if top_feats.get("SupportCalls", 0) > 0 or latest_month.get("SupportCalls", 0) >= 2:
        insights.append("Elevated support call volume suggests unresolved issues — escalate to a senior support rep.")

    if top_feats.get("Complaints", 0) > 0 or latest_month.get("Complaints", 0) >= 1:
        insights.append("Open complaints detected — a satisfaction follow-up could reduce churn risk.")

    if top_feats.get("LoginCount_delta", 0) > 0 or top_feats.get("EngagementScore", 0) > 0:
        insights.append("Declining engagement (logins/session time) — consider a re-engagement campaign or usage tips email.")

    if latest_month.get("ContractType") == "Month-to-Month" and prob >= 0.4:
        insights.append("Customer is on a Month-to-Month contract — offering a discounted annual plan may improve retention.")

    return insights


if __name__ == "__main__":
    sample_history = [
        {"Month": 1, "LoginCount": 25, "SessionDuration": 45, "PurchaseAmount": 80,
         "MonthlyCharge": 55, "DataUsage": 25, "Complaints": 0, "SupportCalls": 0,
         "PaymentDelay": 0, "ContractType": "Month-to-Month", "Tenure": 10, "PlanType": "Standard"},
        {"Month": 2, "LoginCount": 15, "SessionDuration": 30, "PurchaseAmount": 50,
         "MonthlyCharge": 55, "DataUsage": 15, "Complaints": 1, "SupportCalls": 1,
         "PaymentDelay": 3, "ContractType": "Month-to-Month", "Tenure": 11, "PlanType": "Standard"},
        {"Month": 3, "LoginCount": 5, "SessionDuration": 10, "PurchaseAmount": 20,
         "MonthlyCharge": 55, "DataUsage": 5, "Complaints": 2, "SupportCalls": 3,
         "PaymentDelay": 8, "ContractType": "Month-to-Month", "Tenure": 12, "PlanType": "Standard"},
    ]

    predictor = ChurnPredictor()
    output = predictor.predict(sample_history)
    print(json.dumps(output, indent=2))
