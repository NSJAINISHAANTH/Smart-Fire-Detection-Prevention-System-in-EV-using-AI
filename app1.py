# ==============================
# Streamlit Vehicle Prediction
# FIXED VERSION
# ==============================

import streamlit as st
import xgboost as xgb
import joblib
import numpy as np

# -----------------------------
# 1. Load Model
# -----------------------------
model = xgb.Booster()
model.load_model("xgboost_vehicle_model.json")

label_encoder = joblib.load("label_encoder.pkl")

# -----------------------------
# 2. App UI
# -----------------------------
st.title("🚗 Vehicle Health Prediction System")

st.write("Enter vehicle sensor values below.")

battery_voltage = st.number_input(
    "Battery Voltage",
    min_value=0.0,
    max_value=20.0,
    value=12.0
)

temperature = st.number_input(
    "Temperature (°C)",
    min_value=0,
    max_value=150,
    value=30
)

gas_flag = st.selectbox(
    "Gas Detection",
    options=[0, 1],
    format_func=lambda x: "No Gas (0)" if x == 0 else "Gas Detected (1)"
)

# -----------------------------
# 3. Prediction
# -----------------------------
if st.button("Predict"):

    # Create input with feature names (VERY IMPORTANT)
    input_data = np.array([[battery_voltage, temperature, gas_flag]])

    dmatrix = xgb.DMatrix(
        input_data,
        feature_names=["Battery_Voltage", "Temperature", "Gas_Flag"]
    )

    prediction_prob = model.predict(dmatrix)[0]

    prediction_class = 1 if prediction_prob > 0.5 else 0
    prediction_label = label_encoder.inverse_transform([prediction_class])[0]

    st.subheader("Result")

    if prediction_label == "Vehicle Danger":
        st.error("⚠ Vehicle is in DANGER condition!")
    else:
        st.success("✅ Vehicle is in NORMAL condition.")

    st.write(f"Prediction Probability: {prediction_prob:.4f}")