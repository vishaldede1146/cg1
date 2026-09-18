import os
import sys
import shutil
import tempfile
from typing import List, Optional, Union
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from pydantic import BaseModel, Field

# Ensure relative imports resolve correctly
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Initialize APIRouter to mount inside main.py
router = APIRouter()

# Load the real predictor. If model artifacts are missing or loading fails,
# `predictor` stays None and the API returns a 503 instead of a fake response.
_predictor_load_error = None
try:
    import config
    config.ensure_model_artifacts()
    from src.predict import CampaignInput as ModelCampaignInput, get_predictor
    predictor = get_predictor()
except Exception as e:
    predictor = None
    _predictor_load_error = str(e)
    print(f"[email_campaign] Failed to load predictor: {e}")


# --- Pydantic Schemas ---

class CampaignInput(BaseModel):
    campaign_id: Optional[str] = Field(default="CAMP-2026-001", description="Unique campaign reference ID")
    audience_size: int = Field(default=10000, ge=1, description="Total audience target count")
    segment: str = Field(default="Active Customers", description="Audience segment (e.g., 'Active', 'Lapsed', 'VIP')")
    campaign_type: str = Field(default="Promotional", description="Campaign classification (e.g., 'Promotional', 'Newsletter', 'Re-engagement')")
    send_day: int = Field(default=2, ge=0, le=6, description="Day of week (0=Mon, 6=Sun)")
    send_hour: int = Field(default=10, ge=0, le=23, description="Send hour in 24h format")
    discount_percent: float = Field(default=15.0, ge=0.0, le=100.0, description="Offered discount percentage")
    subject: str = Field(default="Exclusive 15% Off Your Next Purchase!", description="Email subject line")
    body: str = Field(default="Hi there! We wanted to offer you an exclusive discount on your favorite products. Click below to redeem.", description="Email body content")
    image_path: Optional[str] = Field(default=None, description="Path or reference to campaign banner image")
    body_length: Optional[int] = Field(default=None, description="Body character count (calculated if missing)")
    num_links: int = Field(default=3, ge=0, description="Number of hyperlinks included in email body")
    num_ctas: int = Field(default=1, ge=0, description="Number of explicit call-to-action buttons")
    has_personalization: int = Field(default=1, ge=0, le=1, description="Flag indicating use of dynamic tags (1=True, 0=False)")
    past_open_rate: float = Field(default=24.5, ge=0.0, le=100.0, description="Historical average open rate (%)")
    past_ctr: float = Field(default=3.2, ge=0.0, le=100.0, description="Historical click-through rate (%)")
    past_conversion_rate: float = Field(default=1.8, ge=0.0, le=100.0, description="Historical conversion rate (%)")
    emails_sent: Optional[int] = Field(default=10000, ge=0)
    emails_delivered: Optional[int] = Field(default=9850, ge=0)
    unique_opens: Optional[int] = Field(default=2413, ge=0)
    unique_clicks: Optional[int] = Field(default=315, ge=0)
    conversions: Optional[int] = Field(default=177, ge=0)
    revenue: Optional[float] = Field(default=5310.0, ge=0.0)
    unsubscribes: Optional[int] = Field(default=12, ge=0)


class TopFactor(BaseModel):
    feature: str
    impact: float
    direction: str  # "Increases Metric" or "Decreases Metric"


class PredictionResponse(BaseModel):
    open_rate: float
    ctr: float
    conversion_rate: float
    roi: float
    unsubscribe_rate: float
    performance_score: float
    top_factors: List[TopFactor]
    business_insights: List[str]
    pipeline_status: str


def _to_model_input(campaign: "CampaignInput") -> "ModelCampaignInput":
    """Converts the API's percentage-scale historical rates (0-100) into the
    0-1 fraction scale the trained model expects (it was trained on rates
    as fractions, not percentages) — without this, predictions saturate."""
    return ModelCampaignInput(
        audience_size=campaign.audience_size,
        segment=campaign.segment,
        campaign_type=campaign.campaign_type,
        send_day=campaign.send_day,
        send_hour=campaign.send_hour,
        discount_percent=campaign.discount_percent,
        subject=campaign.subject,
        body=campaign.body,
        num_links=campaign.num_links,
        num_ctas=campaign.num_ctas,
        has_personalization=campaign.has_personalization,
        past_open_rate=campaign.past_open_rate / 100.0,
        past_ctr=campaign.past_ctr / 100.0,
        past_conversion_rate=campaign.past_conversion_rate / 100.0,
        image_path=campaign.image_path,
    )


def _format_prediction_response(raw: dict, pipeline_status: str) -> dict:
    """Flattens the model's nested {predictions, insights, recommendations}
    output into the flat PredictionResponse schema the frontend expects."""
    preds = raw["predictions"]
    top_features = raw.get("insights", {}).get("top_features", [])
    top_factors = [
        {
            "feature": f["feature"],
            "impact": f["shap_value"],
            "direction": "Increases Metric" if f["direction"] == "increases" else "Decreases Metric",
        }
        for f in top_features
    ]
    return {
        "open_rate": preds["open_rate"],
        "ctr": preds["ctr"],
        "conversion_rate": preds["conversion_rate"],
        "roi": preds["roi"],
        "unsubscribe_rate": preds["unsubscribe_rate"],
        "performance_score": preds["performance_score"],
        "top_factors": top_factors,
        "business_insights": raw.get("recommendations", []),
        "pipeline_status": pipeline_status,
    }


# --- API Routes ---

@router.get("/info")
def get_product_info():
    """Returns micro-product specification and operational pipeline details."""
    return {
        "product_id": "email-campaign-predictor",
        "name": "AI Email Campaign Predictor",
        "version": "1.0.0",
        "inputs": [
            "campaign_id", "audience_size", "segment", "campaign_type", "send_day",
            "send_hour", "discount_percent", "subject", "body", "image_path",
            "body_length", "num_links", "num_ctas", "has_personalization",
            "past_open_rate", "past_ctr", "past_conversion_rate", "emails_sent",
            "emails_delivered", "unique_opens", "unique_clicks", "conversions",
            "revenue", "unsubscribes"
        ],
        "outputs": ["open_rate", "ctr", "conversion_rate", "roi", "unsubscribe_rate", "performance_score", "top_factors", "business_insights"],
        "pipeline": [
            "User Input",
            "Validation",
            "Preprocessing",
            "Multimodal Feature Extraction",
            "Feature Fusion",
            "Multi-Output Email Performance Prediction",
            "SHAP Explainability",
            "Performance Insights"
        ]
    }


@router.get("/health")
def health():
    return {
        "status": "ok" if predictor is not None else "degraded",
        "model_loaded": predictor is not None,
        "load_error": _predictor_load_error,
    }


@router.post("/predict", response_model=PredictionResponse)
def predict(campaign: CampaignInput):
    """Predicts campaign performance metrics from structured payload using the real model only."""
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model is not loaded, so no prediction can be made. Load error: {_predictor_load_error}",
        )
    try:
        raw = predictor.predict(_to_model_input(campaign), explain=True)
        return _format_prediction_response(raw, "Production Multimodal Model Inference")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")


@router.post("/predict-with-image", response_model=PredictionResponse)
async def predict_with_image(
    audience_size: int = Form(10000),
    segment: str = Form("Active Customers"),
    campaign_type: str = Form("Promotional"),
    send_day: int = Form(2),
    send_hour: int = Form(10),
    discount_percent: float = Form(15.0),
    subject: str = Form("Exclusive Discount!"),
    body: str = Form("Check out our new products with special offer."),
    num_links: int = Form(3),
    num_ctas: int = Form(1),
    has_personalization: int = Form(1),
    past_open_rate: float = Form(24.5),
    past_ctr: float = Form(3.2),
    past_conversion_rate: float = Form(1.8),
    image: Optional[UploadFile] = File(None),
):
    """Predicts performance including image asset upload handling."""
    image_path = None
    tmp_dir = None
    try:
        if image is not None:
            tmp_dir = tempfile.mkdtemp()
            image_path = os.path.join(tmp_dir, image.filename)
            with open(image_path, "wb") as f:
                shutil.copyfileobj(image.file, f)

        campaign_input = CampaignInput(
            audience_size=audience_size, segment=segment, campaign_type=campaign_type,
            send_day=send_day, send_hour=send_hour, discount_percent=discount_percent,
            subject=subject, body=body, num_links=num_links, num_ctas=num_ctas,
            has_personalization=has_personalization, past_open_rate=past_open_rate,
            past_ctr=past_ctr, past_conversion_rate=past_conversion_rate,
            image_path=image_path,
        )

        if predictor is None:
            raise HTTPException(
                status_code=503,
                detail=f"Model is not loaded, so no prediction can be made. Load error: {_predictor_load_error}",
            )
        res = predictor.predict(_to_model_input(campaign_input), explain=True)
        return _format_prediction_response(res, "Production Multimodal Model Inference")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_dir and os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)