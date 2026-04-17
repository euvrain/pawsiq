from fastapi import APIRouter, Query
from typing import Optional
import pandas as pd
from pathlib import Path

router = APIRouter()
DATA = Path("/app/data/synthetic")

@router.get("")
def list_bookings(status: Optional[str]=None, limit: int=Query(50, le=500)):
    df = pd.read_csv(DATA / "bookings.csv", parse_dates=["scheduled_at"])
    if status:
        df = df[df["status"] == status]
    df = df.sort_values("scheduled_at", ascending=False).head(limit)
    return {"bookings": df.to_dict(orient="records"), "count": len(df)}

@router.get("/summary")
def booking_summary():
    df        = pd.read_csv(DATA / "bookings.csv", parse_dates=["scheduled_at"])
    completed = df[df["status"] == "completed"]
    avg_surge = float(completed["surge_multiplier"].mean())
    completed["year_month"] = completed["scheduled_at"].dt.to_period("M").astype(str)
    monthly = completed.groupby("year_month")["final_price"].sum().tail(6).round(2).to_dict()
    return {
        "total_bookings":      len(df),
        "completed_bookings":  len(completed),
        "completion_rate_pct": round(len(completed)/len(df)*100, 1),
        "total_revenue":       round(float(completed["final_price"].sum()), 2),
        "avg_surge":           round(avg_surge, 4),
        "peak_hour_pct":       round(float(completed["is_peak_hour"].mean()*100), 1),
        "peak_avg_price":      round(float(completed[completed["is_peak_hour"]==1]["final_price"].mean()), 2),
        "offpeak_avg_price":   round(float(completed[completed["is_peak_hour"]==0]["final_price"].mean()), 2),
        "revenue_lift_vs_flat_pct": round((avg_surge-1)*100, 1),
        "monthly_revenue":     monthly,
    }
