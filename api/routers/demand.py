from fastapi import APIRouter
from pydantic import BaseModel, Field
import numpy as np
from api.ml_models import demand_model, build_demand_features

router = APIRouter()

class DemandRequest(BaseModel):
    hour_of_day: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6)
    month:       int = Field(..., ge=1, le=12)
    year:        int = Field(2025)

class DemandResponse(BaseModel):
    predicted_bookings: float
    demand_level:       str
    is_peak_hour:       bool
    hour_of_day:        int
    day_of_week:        int

DOW = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

@router.post("/demand", response_model=DemandResponse)
def predict_demand(req: DemandRequest):
    features  = build_demand_features(req.hour_of_day, req.day_of_week, req.month, req.year)
    raw       = float(demand_model.predict(features)[0])
    predicted = round(max(0.0, raw), 2)
    level     = "high" if predicted>=2.5 else ("medium" if predicted>=1.5 else "low")
    return DemandResponse(
        predicted_bookings=predicted, demand_level=level,
        is_peak_hour=req.hour_of_day in [7,8,9,17,18,19],
        hour_of_day=req.hour_of_day, day_of_week=req.day_of_week,
    )

@router.post("/demand/heatmap")
def demand_heatmap(month: int = 4, year: int = 2025):
    results = []
    for dow in range(7):
        for hour in range(6, 21):
            features  = build_demand_features(hour, dow, month, year)
            predicted = round(max(0.0, float(demand_model.predict(features)[0])), 2)
            results.append({"hour_of_day":hour,"day_of_week":dow,
                "day_label":DOW[dow],"predicted":predicted,
                "is_peak":hour in [7,8,9,17,18,19]})
    return {"heatmap": results, "month": month, "year": year}
