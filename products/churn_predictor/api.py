import os
import sys
from typing import List, Optional, Union
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

# Ensure relative imports resolve correctly
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Initialize APIRouter to mount inside main.py
router = APIRouter()

# Load the real predictor. If model artifacts are missing or loading fails,
# `_predictor` stays None and the API returns a 503 instead of a fake response.
_predictor_load_error = None
try:
    from churn_src.predict_pipeline import ChurnPredictor, ValidationError
    _predictor = ChurnPredictor()
except Exception as e:
    _predictor = None
    _predictor_load_error = str(e)
    print(f"[churn_predictor] Failed to load predictor: {e}")


# --- Pydantic Schemas ---

class CustomerRecord(BaseModel):
    CustomerID: Optional[str] = Field(default="CUST-1001", description="Unique customer reference ID")
    Month: int = Field(default=12, ge=1, description="Sequential account month")
    LoginCount: float = Field(default=8.0, ge=0.0, description="Monthly application logins")
    SessionDuration: float = Field(default=45.0, ge=0.0, description="Average session time in minutes")
    PurchaseAmount: float = Field(default=120.0, ge=0.0, description="Total monthly spend ($)")
    MonthlyCharge: float = Field(default=49.99, ge=0.0, description="Subscription plan monthly price ($)")
    DataUsage: float = Field(default=15.5, ge=0.0, description="Data or bandwidth consumed (GB)")
    Complaints: float = Field(default=1.0, ge=0.0, description="Log of registered complaints")
    SupportCalls: float = Field(default=2.0, ge=0.0, description="Support ticket contact count")
    PaymentDelay: float = Field(default=3.0, ge=0.0, description="Days payment was delayed")
    ContractType: str = Field(default="Month-to-Month", description="Contract duration ('Month-to-Month', 'One Year', 'Two Year')")
    Tenure: float = Field(default=6.0, ge=0.0, description="Total customer tenure in months")
    PlanType: str = Field(default="Standard", description="Plan tier ('Basic', 'Standard', 'Enterprise')")


class PredictionRequest(BaseModel):
    customer_id: Optional[str] = "CUST-1001"
    history: Optional[List[CustomerRecord]] = None
    single_record: Optional[CustomerRecord] = None
    threshold: float = Field(0.5, ge=0.0, le=1.0)


class TopFactor(BaseModel):
    feature: str
    impact: float
    direction: str  # "Increases Churn Risk" or "Decreases Churn Risk"


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: int
    risk_level: str
    top_factors: List[TopFactor]
    business_insights: List[str]
    pipeline_status: str


# --- API Routes ---

@router.get("/info")
def get_product_info():
    """Returns micro-product specification and operational pipeline details."""
    return {
        "product_id": "churn-predictor",
        "name": "AI Customer Churn Predictor",
        "version": "1.0.0",
        "inputs": [
            "CustomerID", "Month", "LoginCount", "SessionDuration", 
            "PurchaseAmount", "MonthlyCharge", "DataUsage", "Complaints", 
            "SupportCalls", "PaymentDelay", "ContractType", "Tenure", "PlanType"
        ],
        "outputs": ["churn_probability", "churn_prediction", "risk_level", "top_factors", "business_insights"],
        "pipeline": [
            "Customer Input",
            "Input Validation",
            "Data Cleaning",
            "Feature Engineering",
            "Sequence Preparation",
            "Scaling",
            "LSTM Prediction",
            "SHAP Explainability",
            "Business Insights",
            "Final Output"
        ]
    }


@router.get("/health")
def health():
    return {
        "status": "ok" if _predictor is not None else "degraded",
        "model_loaded": _predictor is not None,
        "load_error": _predictor_load_error,
    }


@router.post("/predict", response_model=PredictionResponse)
def predict(request: Union[PredictionRequest, CustomerRecord]):
    """Calculates churn risk using either real trained model artifacts or deterministic rule pipeline."""
    # Handle direct single record payloads vs wrapped PredictionRequest payloads
    if isinstance(request, CustomerRecord):
        target_record = request
        threshold = 0.5
    else:
        if request.single_record:
            target_record = request.single_record
        elif request.history and len(request.history) > 0:
            target_record = request.history[-1]
        else:
            target_record = CustomerRecord()
        threshold = request.threshold

    # Use the real ML model only. No rule-based fallback.
    if _predictor is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model is not loaded, so no prediction can be made. Load error: {_predictor_load_error}",
        )
    try:
        input_dict = target_record.model_dump()
        res = _predictor.predict([input_dict], explain=True, threshold=threshold)
        res["pipeline_status"] = "Production LSTM Inference"
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")