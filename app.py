import streamlit as st
import numpy as np
import joblib
import plotly.graph_objects as go
import math

# ---- Page Configuration ----
st.set_page_config(
    page_title="📊 Customer Churn Predictor",
    page_icon="📈",
    layout="centered"
)

# ---- Load Model ----
model_data = joblib.load("models/xgb_model.pkl")
model = model_data["model"]
feature_names = model_data["features"]

# ---- App Title ----
st.markdown("## 📊 Customer Churn Prediction")
st.write("Enter customer details below to predict the likelihood of churn.")

# ---- Input Form ----
st.subheader("👤 Customer Information")
col1, col2 = st.columns(2)

with col1:
    customer_age = st.number_input("Age", 18, 100, 40)
    gender = st.radio("Gender", ["Female", "Male"])
    dependent_count = st.number_input("Dependents", 0, 10, 1)
    months_on_book = st.number_input("Months on Book", 1, 100, 36)
    total_relationship_count = st.number_input("Total Relationship Count", 1, 10, 3)
    months_inactive_12_mon = st.number_input("Inactive Months (12M)", 0, 12, 2)
    contacts_count_12_mon = st.number_input("Contacts Count (12M)", 0, 20, 3)

with col2:
    credit_limit = st.number_input("💳 Credit Limit", 0.0, 100000.0, 5000.0, step=100.0)
    total_revolving_bal = st.number_input("🔁 Total Revolving Balance", 0.0, 100000.0, 1000.0, step=100.0)
    avg_open_to_buy = st.number_input("Average Open To Buy", 0.0, 100000.0, 4000.0, step=100.0)
    total_amt_chng_q4_q1 = st.number_input("Amount Change Q4→Q1", 0.0, 10.0, 1.5, step=0.1)
    total_trans_amt = st.number_input("Total Transaction Amount", 0.0, 100000.0, 5000.0, step=100.0)
    total_trans_ct = st.number_input("Total Transaction Count", 0, 500, 50)

# ---- Encode Gender ----
gender_val = 1 if gender == "Female" else 0

# ---- Prediction & Needle Gauge ----
if st.button("🔮 Predict Churn"):
    features = np.array([
        customer_age, gender_val, dependent_count, months_on_book,
        total_relationship_count, months_inactive_12_mon, contacts_count_12_mon,
        credit_limit, total_revolving_bal, avg_open_to_buy,
        total_amt_chng_q4_q1, total_trans_amt, total_trans_ct
    ]).reshape(1, -1)

    probability = model.predict_proba(features)[0][1]
    prediction_label = "Churn" if probability >= 0.5 else "No Churn"

    # ---- Display Prediction Text ----
    if prediction_label == "Churn":
        st.error(f"⚠️ Customer is **likely to churn** (Probability: {probability:.2%})")
    else:
        st.success(f"✅ Customer is **not likely to churn** (Probability: {probability:.2%})")

    # ---- Semicircle Needle Gauge ----
    sectors = ["Low", "Medium", "High", "Extreme"]
    colors = ["green", "blue", "yellow", "red"]
    values = [0.25, 0.25, 0.25, 0.25]  # sum = 1 (normalized for plotly pie)

    fig = go.Figure()

    # Pie sectors (semicircle)
    fig.add_trace(go.Pie(
        values=values + [1],  # extra part to hide bottom half
        rotation=90,
        hole=0.5,
        marker_colors=colors + ["white"],
        text=sectors + [""],
        textinfo="text",
        direction="clockwise",
        showlegend=False
    ))

    # ---- Needle Calculation ----
    theta = 180 * probability  # 0-1 mapped to 0-180 degrees
    radians = math.radians(180 - theta)
    needle_length = 0.2

    x_center, y_center = 0.5, 0.5
    x_head = x_center + needle_length * math.cos(radians)
    y_head = y_center + needle_length * math.sin(radians)

    fig.add_shape(
        type="line",
        x0=x_center, y0=y_center,
        x1=x_head, y1=y_head,
        line=dict(color="black", width=4)
    )

    fig.update_layout(
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=False,
        height=400,
        annotations=[dict(
            x=0.5, y=0.05,
            text=f"Probability: {probability:.2%}",
            showarrow=False,
            font=dict(size=20, color="black")
        )]
    )

    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("💡 Make sure the model and preprocessing match the training pipeline. Run this app using `streamlit run app.py`.")
