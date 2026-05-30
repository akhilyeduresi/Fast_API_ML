from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle

app = FastAPI()

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

class HouseInput(BaseModel):
    rooms: int
    size: int
    age: int

@app.get("/")
def home():
    return {"message": "ML Model is running inside Docker!"}

@app.post("/predict")
def predict(data: HouseInput):
    features = np.array([[data.rooms, data.size, data.age]])
    prediction = model.predict(features)[0]
    return {"prediction": round(prediction, 2)}