import pickle
import pandas as pd
import numpy as np
from pathlib import Path

ROOT      = Path(__file__).parent.parent
ARTIFACTS = ROOT / "ml" / "artifacts"

with open(ARTIFACTS / "demand_model.pkl", "rb") as f:
    demand_model = pickle.load(f)
demand_hist_avg = pd.read_csv(ARTIFACTS / "demand_hist_avg.csv")

with open(ARTIFACTS / "pricing_model.pkl", "rb") as f:
    pricing_bundle = pickle.load(f)
pricing_model   = pricing_bundle["model"]
pricing_scaler  = pricing_bundle["scaler"]
pricing_hist    = pd.read_csv(ARTIFACTS / "pricing_hist_avg.csv")
PRICING_FEATURES = pricing_bundle["features"]

DEMAND_FEATURES = [
    "hour_of_day","day_of_week","month","year",
    "is_weekend","is_peak_hour","is_morning","is_evening",
    "hour_sin","hour_cos","dow_sin","dow_cos","month_sin","month_cos",
    "hist_avg_bookings",
]

def build_demand_features(hour, dow, month, year):
    mask = ((demand_hist_avg["hour_of_day"]==hour)&(demand_hist_avg["day_of_week"]==dow))
    hist = float(demand_hist_avg.loc[mask,"hist_avg_bookings"].values[0]) if mask.any() else 0.85
    return pd.DataFrame([{
        "hour_of_day":hour,"day_of_week":dow,"month":month,"year":year,
        "is_weekend":int(dow>=5),"is_peak_hour":int(hour in [7,8,9,17,18,19]),
        "is_morning":int(hour in [7,8,9]),"is_evening":int(hour in [17,18,19]),
        "hour_sin":np.sin(2*np.pi*hour/24),"hour_cos":np.cos(2*np.pi*hour/24),
        "dow_sin":np.sin(2*np.pi*dow/7),"dow_cos":np.cos(2*np.pi*dow/7),
        "month_sin":np.sin(2*np.pi*month/12),"month_cos":np.cos(2*np.pi*month/12),
        "hist_avg_bookings":hist,
    }])[DEMAND_FEATURES]

def build_pricing_features(hour, dow, month, base_price, svc, zip_demand):
    mask = ((pricing_hist["hour_of_day"]==hour)&(pricing_hist["day_of_week"]==dow))
    hist = float(pricing_hist.loc[mask,"hist_avg_surge"].values[0]) if mask.any() else 1.18
    return pd.DataFrame([{
        "hour_of_day":hour,"day_of_week":dow,"month":month,
        "is_peak_hour":int(hour in [7,8,9,17,18,19]),"is_weekend":int(dow>=5),
        "hour_sin":np.sin(2*np.pi*hour/24),"hour_cos":np.cos(2*np.pi*hour/24),
        "dow_sin":np.sin(2*np.pi*dow/7),"dow_cos":np.cos(2*np.pi*dow/7),
        "month_sin":np.sin(2*np.pi*month/12),"month_cos":np.cos(2*np.pi*month/12),
        "zip_hour_demand":zip_demand,"hist_avg_surge":hist,"base_price":base_price,
        "svc_walk_30":int(svc=="walk_30"),"svc_overnight":int(svc=="overnight"),
        "svc_walk_60":int(svc=="walk_60"),
    }])[PRICING_FEATURES]
