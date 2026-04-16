from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import Literal
import numpy as np
from api.ml_models import pricing_model, pricing_scaler, build_pricing_features

router = APIRouter()

BASE_PRICES = {"walk_30":16.00,"walk_60":24.00,"drop_in":14.00,"overnight":55.00}
ServiceType = Literal["walk_30","walk_60","drop_in","overnight"]

class PricingRequest(BaseModel):
    hour_of_day:     int         = Field(..., ge=0, le=23)
    day_of_week:     int         = Field(..., ge=0, le=6)
    month:           int         = Field(..., ge=1, le=12)
    service_type:    ServiceType = "walk_30"
    zip_hour_demand: int         = Field(1, ge=1, le=20)

class PricingResponse(BaseModel):
    surge_multiplier:  float
    base_price:        float
    final_price:       float
    pricing_tier:      str
    is_peak_hour:      bool
    service_type:      str
    price_explanation: str

@router.post("/price", response_model=PricingResponse)
def predict_price(req: PricingRequest):
    base       = BASE_PRICES[req.service_type]
    features   = build_pricing_features(req.hour_of_day, req.day_of_week,
                    req.month, base, req.service_type, req.zip_hour_demand)
    features_sc= pricing_scaler.transform(features)
    raw        = float(pricing_model.predict(features_sc)[0])
    surge      = round(float(np.clip(raw, 0.85, 1.35)), 4)
    final      = round(base * surge, 2)
    tier       = "high" if surge>=1.25 else ("medium" if surge>=1.10 else "standard")
    is_peak    = req.hour_of_day in [7,8,9,17,18,19]
    explanation= f"Peak hour — {int((surge-1)*100)}% surge" if surge>1.15 and is_peak else "Standard pricing"
    return PricingResponse(surge_multiplier=surge, base_price=base, final_price=final,
        pricing_tier=tier, is_peak_hour=is_peak, service_type=req.service_type,
        price_explanation=explanation)

@router.get("/price/schedule")
def pricing_schedule(month: int = 4):
    results = []
    for dow in range(7):
        for hour in range(6, 21):
            for svc, base in BASE_PRICES.items():
                features    = build_pricing_features(hour, dow, month, base, svc, 1)
                features_sc = pricing_scaler.transform(features)
                raw         = float(pricing_model.predict(features_sc)[0])
                surge       = round(float(np.clip(raw, 0.85, 1.35)), 4)
                results.append({"hour_of_day":hour,"day_of_week":dow,
                    "service_type":svc,"surge":surge,"final_price":round(base*surge,2)})
    return {"schedule": results, "month": month}
