# ==========================================
# File : predict.py
# Purpose : Load Model & Predict Iris Species
# ==========================================

import os
import joblib
import pandas as pd

# -----------------------------------------
# Load Model
# -----------------------------------------

CURRENT_FOLDER = os.path.dirname(__file__)

MODEL_PATH = os.path.join(
    CURRENT_FOLDER,
    "..",
    "2_Models",
    "iris_model.pkl"
)

model = joblib.load(MODEL_PATH)


# -----------------------------------------
# Prediction Function
# -----------------------------------------

def predict_species(
    sepal_length,
    sepal_width,
    petal_length,
    petal_width
):

    sample = pd.DataFrame({

        "SepalLengthCm": [sepal_length],

        "SepalWidthCm": [sepal_width],

        "PetalLengthCm": [petal_length],

        "PetalWidthCm": [petal_width]

    })

    prediction = model.predict(sample)

    return prediction[0]


# -----------------------------------------
# Test
# -----------------------------------------

if __name__ == "__main__":

    flower = predict_species(
        5.1,
        3.5,
        1.4,
        0.2
    )

    print("Prediction :", flower)