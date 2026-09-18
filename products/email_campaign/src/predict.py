"""
Prediction pipeline for a single new campaign:

    User Input -> Validation -> Preprocessing -> Multimodal Feature
    Extraction -> Feature Fusion -> Multi-Output Prediction ->
    SHAP Explainability -> Performance Insights

This module is imported by both the Streamlit app and the FastAPI
service so the exact same logic powers both interfaces.
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd
import shap
import torch
from PIL import Image
from pydantic import BaseModel, Field, field_validator

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.dataset import IMAGE_TRANSFORM
from src.evaluate import load_model
from src.data_preprocessing import engineer_features
from transformers import AutoTokenizer


# ----------------------------------------------------------------------
# 1. Input schema + validation
# ----------------------------------------------------------------------
class CampaignInput(BaseModel):
    audience_size: int = Field(..., gt=0, le=2_000_000)
    segment: str
    campaign_type: str
    send_day: int = Field(..., ge=0, le=6)
    send_hour: int = Field(..., ge=0, le=23)
    discount_percent: float = Field(..., ge=0, le=100)
    subject: str
    body: str
    num_links: int = Field(..., ge=0, le=50)
    num_ctas: int = Field(..., ge=0, le=20)
    has_personalization: int = Field(..., ge=0, le=1)
    past_open_rate: float = Field(..., ge=0, le=1)
    past_ctr: float = Field(..., ge=0, le=1)
    past_conversion_rate: float = Field(..., ge=0, le=1)
    image_path: str | None = None  # path to an uploaded creative image, optional

    @field_validator("segment")
    @classmethod
    def validate_segment(cls, v):
        if v not in config.SEGMENTS:
            raise ValueError(f"segment must be one of {config.SEGMENTS}")
        return v

    @field_validator("campaign_type")
    @classmethod
    def validate_campaign_type(cls, v):
        if v not in config.CAMPAIGN_TYPES:
            raise ValueError(f"campaign_type must be one of {config.CAMPAIGN_TYPES}")
        return v

    @field_validator("subject", "body")
    @classmethod
    def non_empty_text(cls, v):
        if not v or not v.strip():
            raise ValueError("Text fields must not be empty")
        return v


# ----------------------------------------------------------------------
# 2. Predictor class wrapping the full pipeline
# ----------------------------------------------------------------------
class CampaignPredictor:
    def __init__(self, device=None):
        self.device = device or config.DEVICE
        self.model, self.checkpoint = load_model(device=self.device)
        self.preprocessor = joblib.load(config.PREPROCESSOR_PATH)
        self.tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)
        self.feature_names = self.preprocessor.feature_names()

    # -- preprocessing -----------------------------------------------
    def _row_from_input(self, campaign_input: CampaignInput) -> pd.DataFrame:
        row = campaign_input.model_dump()
        row.pop("image_path", None)
        df = pd.DataFrame([row])
        df = engineer_features(df)
        return df

    def _load_image_tensor(self, image_path):
        if image_path and os.path.exists(image_path):
            img = Image.open(image_path).convert("RGB")
        else:
            img = Image.new("RGB", (config.IMAGE_SIZE, config.IMAGE_SIZE), color=(128, 128, 128))
        return IMAGE_TRANSFORM(img).unsqueeze(0)

    def _tokenize(self, subject, body):
        text = f"{subject} [SEP] {body}"
        encoded = self.tokenizer(
            text, padding="max_length", truncation=True,
            max_length=config.TEXT_MAX_LENGTH, return_tensors="pt",
        )
        return encoded["input_ids"], encoded["attention_mask"]

    # -- full pipeline -------------------------------------------------
    @torch.no_grad()
    def predict(self, campaign_input: CampaignInput, explain=True, top_k_features=8):
        # Validation happens automatically via Pydantic when CampaignInput is constructed.
        df = self._row_from_input(campaign_input)
        tabular_np = self.preprocessor.transform(df)
        tabular = torch.tensor(tabular_np, dtype=torch.float32).to(self.device)

        input_ids, attention_mask = self._tokenize(campaign_input.subject, campaign_input.body)
        input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)

        image = self._load_image_tensor(campaign_input.image_path).to(self.device)

        self.model.eval()
        stacked_pred, per_target = self.model(tabular, input_ids, attention_mask, image)
        stacked_pred = stacked_pred.cpu().numpy()[0]

        predictions = {name: float(val) for name, val in zip(config.TARGET_COLUMNS, stacked_pred)}
        predictions = self._postprocess(predictions)

        result = {"predictions": predictions}

        if explain:
            result["insights"] = self.explain(df, tabular_np, input_ids, attention_mask, image,
                                                top_k_features=top_k_features)
            result["recommendations"] = self.generate_recommendations(predictions, campaign_input)

        return result

    def _postprocess(self, predictions):
        """Clip to sane display ranges and round for readability."""
        predictions["open_rate"] = round(float(np.clip(predictions["open_rate"], 0, 1)) * 100, 2)
        predictions["ctr"] = round(float(np.clip(predictions["ctr"], 0, 1)) * 100, 2)
        predictions["conversion_rate"] = round(float(np.clip(predictions["conversion_rate"], 0, 1)) * 100, 2)
        predictions["roi"] = round(float(predictions["roi"]) * 100, 2)
        predictions["unsubscribe_rate"] = round(float(np.clip(predictions["unsubscribe_rate"], 0, 1)) * 100, 2)
        predictions["performance_score"] = round(float(np.clip(predictions["performance_score"], 0, 100)), 2)
        return predictions

    # -- SHAP explainability for a single instance ----------------------
    def explain(self, df, tabular_np, input_ids, attention_mask, image, top_k_features=8,
                target="performance_score", nsamples=80):
        target_idx = config.TARGET_COLUMNS.index(target)

        with torch.no_grad():
            text_emb = self.model.text_encoder(input_ids, attention_mask)
            img_emb = self.model.image_encoder(image)

        def f(X):
            with torch.no_grad():
                X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
                tab_emb = self.model.tabular_encoder(X_t)
                n = tab_emb.shape[0]
                text_rep = text_emb.repeat(n, 1)
                img_rep = img_emb.repeat(n, 1)
                fused = torch.cat([tab_emb, text_rep, img_rep], dim=1)
                trunk_out = self.model.fusion_trunk(fused)
                pred = self.model.heads[target](trunk_out).squeeze(-1)
            return pred.cpu().numpy()

        # Background: small synthetic neighborhood around the instance (mean-zero jitter)
        # so we avoid depending on having the training set in memory at inference time.
        rng = np.random.default_rng(config.RANDOM_SEED)
        background = tabular_np + rng.normal(0, 0.5, size=(30, tabular_np.shape[1]))

        explainer = shap.KernelExplainer(f, background, silent=True)
        shap_values = np.array(explainer.shap_values(tabular_np, nsamples=nsamples, silent=True)).reshape(-1)

        feature_importance = sorted(
            zip(self.feature_names, shap_values), key=lambda x: abs(x[1]), reverse=True
        )[:top_k_features]

        return {
            "target_explained": target,
            "top_features": [
                {"feature": fname, "shap_value": round(float(val), 5),
                 "direction": "increases" if val > 0 else "decreases"}
                for fname, val in feature_importance
            ],
        }

    # -- simple rule-based recommendations layered on top of predictions --
    def generate_recommendations(self, predictions, campaign_input: CampaignInput):
        tips = []
        if predictions["open_rate"] < 15:
            tips.append("Predicted open rate is low — consider a more curiosity-driven subject line "
                         "or sending at a higher-engagement hour (e.g. 8-9am or 6-7pm).")
        if campaign_input.has_personalization == 0:
            tips.append("Adding personalization (e.g. first name) tends to lift open rates.")
        if predictions["ctr"] < 2:
            tips.append("Predicted CTR is low — try reducing the number of competing links and "
                         "using a single, clear primary CTA.")
        if campaign_input.num_links > 6:
            tips.append("This email has many links, which can dilute click-through — "
                         "consider trimming to 2-4 focused links.")
        if predictions["unsubscribe_rate"] > 1.0:
            tips.append("Predicted unsubscribe rate is elevated — a very high discount or aggressive "
                         "framing can trigger fatigue; consider moderating the offer or frequency.")
        if predictions["roi"] < 0:
            tips.append("Predicted ROI is negative at this discount level — consider a smaller "
                         "discount or targeting a higher-intent segment.")
        if not tips:
            tips.append("This campaign configuration looks solid based on historical patterns.")
        return tips


# ----------------------------------------------------------------------
# Convenience singleton loader (avoids reloading model weights repeatedly)
# ----------------------------------------------------------------------
_predictor_instance = None


def get_predictor():
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = CampaignPredictor()
    return _predictor_instance
