# ==========================================
# File : app.py
# Purpose : FastAPI Application
# ==========================================
# FastAPI Application for Iris Prediction

from fastapi import FastAPI
from fastapi import FastAPI
from pydantic import BaseModel

from predict import predict_species


app = FastAPI(
    title="IRIS Flower Classification API",
    version="1.0"
)


# -----------------------------------------
# Input Schema
# -----------------------------------------

class IrisInput(BaseModel):

    sepal_length: float

    sepal_width: float

    petal_length: float

    petal_width: float


# -----------------------------------------
# Home API
# -----------------------------------------

@app.get("/")
def home():

    return {
        "message": "Welcome to IRIS Flower Classification API"
    }


# -----------------------------------------
# Prediction API
# -----------------------------------------

@app.post("/predict")
def predict(data: IrisInput):

    result = predict_species(

        data.sepal_length,

        data.sepal_width,

        data.petal_length,

        data.petal_width

    )

    return {

        "Predicted Species": result

    }