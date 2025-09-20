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
model_data = joblib.load("models/xgb_model_v2.pkl")
model = model_data["model"]
feature_names = model_data["features"]

# ---- Cost Definitions ----
FN_COST = 10000  # Cost of losing a churner (False Negative)
FP_COST = 500    # Cost of unnecessary retention offer (False Positive)

# ---- App Title ----
st.markdown("## 📊 Customer Churn Prediction with Cost Analysis")
st.write("Enter customer details below to predict churn probability and estimated cost impact.")

# ---- Decision Threshold ----
threshold = st.slider("🔧 Decision Threshold", 0.1, 0.9, 0.5, 0.01)

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
    total_ct_chng_q4_q1 = st.number_input("Transaction Count Change Q4→Q1", 0.0, 5.0, 1.0, step=0.1)
    avg_utilization_ratio = st.number_input("Avg Utilization Ratio", 0.0, 1.0, 0.5, step=0.01)

# ---- Encode Gender ----
gender_val = 1 if gender == "Female" else 0

# ---- Prediction & Cost Calculation ----
if st.button("🔮 Predict Churn & Cost"):
    features = np.array([
        customer_age, gender_val, dependent_count, months_on_book,
        total_relationship_count, months_inactive_12_mon, contacts_count_12_mon,
        credit_limit, total_revolving_bal, avg_open_to_buy,
        total_amt_chng_q4_q1, total_trans_amt, total_trans_ct,
        total_ct_chng_q4_q1, avg_utilization_ratio
    ]).reshape(1, -1)

    # Validate feature length
    if features.shape[1] != len(feature_names):
        st.error(f"❌ Feature mismatch: expected {len(feature_names)}, got {features.shape[1]}")
    else:
        probability = model.predict_proba(features)[0][1]
        prediction_label = "Churn" if probability >= threshold else "No Churn"

        # ---- Cost Calculation ----
        estimated_cost = FP_COST * (1 - probability) if prediction_label == "Churn" else FN_COST * probability

        # ---- Display Prediction ----
        if prediction_label == "Churn":
            st.error(f"⚠️ Likely to churn (Prob: {probability:.2%})")
        else:
            st.success(f"✅ Unlikely to churn (Prob: {probability:.2%})")

        st.info(f"💰 **Estimated Cost Impact:** ₹{estimated_cost:,.0f}")

        # ---- Gauge Chart ----
        sectors = ["Low", "Medium", "High", "Extreme"]
        colors = ["green", "blue", "yellow", "red"]
        values = [0.25, 0.25, 0.25, 0.25]

        fig = go.Figure(go.Pie(
            values=values + [1],
            rotation=90,
            hole=0.5,
            marker_colors=colors + ["white"],
            text=sectors + [""],
            textinfo="text",
            showlegend=False
        ))

        theta = 180 * probability
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
                text=f"Prob: {probability:.2%}",
                showarrow=False,
                font=dict(size=20)
            )]
        )

        st.plotly_chart(fig, use_container_width=True)
with st.expander("ℹ️ Learn about Cost Impact"):
    st.write(
        """
        This is the expected financial risk for this prediction:  
        - **Churn Prediction:** Money wasted if you offer retention unnecessarily.  
        - **No Churn Prediction:** Revenue lost if you miss a real churner.  
        Based on FN = ₹10,000 and FP = ₹500.
        """
    )

st.markdown("---")
st.caption("💡 Adjust threshold to balance false positives/negatives and cost impact.")
