import joblib
import pandas as pd
from src.preprocessing import preprocess

# Load model once at import time
model = joblib.load("models/noshow_model.pkl")

def predict(df: pd.DataFrame):
    """
    Takes a raw appointment DataFrame, preprocesses it,
    and returns no-show risk probabilities.
    """
    X = preprocess(df)
    probs = model.predict_proba(X)[:, 1]
    return probs

