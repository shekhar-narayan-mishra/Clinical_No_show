import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import joblib
from src.predict import predict

# Load trained model
model = joblib.load("models/noshow_model.pkl")

st.title("Clinical Appointment No-Show Prediction System")

uploaded_file = st.file_uploader("Upload Appointment CSV")

if uploaded_file:

    # Read uploaded data
    df = pd.read_csv(uploaded_file)

    # Get prediction probabilities
    probs = predict(df)

    # Add predictions
    df["No_show_risk_probability"] = probs

    # Risk categorization
    df["Risk_Level"] = pd.cut(
        df["No_show_risk_probability"],
        bins=[0, 0.3, 0.6, 1],
        labels=["Low", "Medium", "High"]
    )

    # -----------------------
    # Prediction Output
    # -----------------------
    st.subheader("Prediction Output")
    st.dataframe(df[["No_show_risk_probability", "Risk_Level"]])

    # -----------------------
    # Risk Summary
    # -----------------------
    st.subheader("Risk Summary")

    risk_counts = df["Risk_Level"].value_counts().sort_index()
    st.bar_chart(risk_counts)

    # -----------------------
    # Feature Importance
    # -----------------------
    st.subheader("Model Feature Importance")

    importance = model.feature_importances_


    feature_names = model.feature_names_in_

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    }).sort_values(by="Importance", ascending=False)

    st.bar_chart(importance_df.set_index("Feature"))

    st.success("Prediction completed successfully!")