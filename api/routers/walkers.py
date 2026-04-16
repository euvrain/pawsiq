from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pandas as pd
from pathlib import Path

router = APIRouter()
DATA = Path(__file__).parent.parent.parent / "data" / "synthetic"

class WalkerResponse(BaseModel):
    user_id:     str
    name:        str
    email:       str
    zip:         str
    rating:      float
    total_walks: int

@router.get("", response_model=list[WalkerResponse])
def list_walkers():
    df = pd.read_csv(DATA / "walkers.csv")
    df["zip"] = df["zip"].astype(str)
    return df.to_dict(orient="records")

@router.get("/{walker_id}", response_model=WalkerResponse)
def get_walker(walker_id: str):
    df     = pd.read_csv(DATA / "walkers.csv")
    df["zip"] = df["zip"].astype(str)
    walker = df[df["user_id"] == walker_id]
    if walker.empty:
        raise HTTPException(status_code=404, detail=f"Walker {walker_id} not found")
    return walker.iloc[0].to_dict()
