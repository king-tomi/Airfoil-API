from fastapi import FastAPI
from joblib import load
import numpy as np
from pydantic import BaseModel

api = FastAPI()

class AirFoil(BaseModel):
    frequency: float
    angle: float
    chord_length: float
    velocity: float
    suction: float


model = load("rf_model.pkl")



@api.get("/")
async def home():
    return {"message": "get predictions using /prediction"}

@api.get("/prediction/")
async def predict(data: AirFoil):
    df = data.dict()
    arr = np.array(df.values()).reshape(1,-1)
    pred = model.predict(arr)
    return {"prediction": pred}