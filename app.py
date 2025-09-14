import streamlit as st
import numpy as np
import joblib

# Load model and feature names
model_data = joblib.load("models/xgbModel.pkl")
model = model_data["model"]
features = model_data["features"]

st.set_page_config(page_title="Churn Predictor", page_icon="📊")
st.title("📊 Customer Churn Prediction")
st.write("Choose the customer features below and click **Predict**.")

# Collect user input
user_input = []
for feat in features:
    if feat.lower() in ["tenure", "monthlycharges", "totalcharges"]:
        val = st.number_input(f"{feat}", min_value=0.0, step=1.0)
    else:
        checked = st.checkbox(feat, value=False)
        val = 1 if checked else 0
    user_input.append(val)

if st.button("Predict"):
    X = np.array(user_input).reshape(1, -1)
    pred_prob = model.predict_proba(X)[0][1]
    label = "Churn" if pred_prob >= 0.5 else "No Churn"
    st.subheader(f"Prediction: **{label}**")
    st.write(f"Probability of churn: {pred_prob:.2%}")
