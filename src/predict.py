import joblib
from src.preprocessing import preprocess

model = joblib.load("models/noshow_model.pkl")

def predict(df):

    X = preprocess(df)
    probs = model.predict_proba(X)[:,1]

    return probs
