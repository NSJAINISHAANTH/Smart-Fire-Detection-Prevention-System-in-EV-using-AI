# ===============================================
# COMBINED MULTI-MODEL STREAMLIT APPLICATION
# ===============================================

import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.preprocessing import LabelEncoder

# -----------------------------------------------
# PAGE CONFIG
# -----------------------------------------------
st.set_page_config(
    page_title="Battery & Vehicle Prediction System",
    layout="centered"
)

st.title("🔮 Multi Model Prediction System")

# -----------------------------------------------
# SIDEBAR NAVIGATION
# -----------------------------------------------
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox(
    "Select Prediction Model",
    [
        "BV Temperature Prediction",
        "Vehicle Health Prediction",
        "Battery Health Prediction"
    ]
)

# ===============================================
# LOAD MODELS (CACHED)
# ===============================================

@st.cache_resource
def load_temperature_model():
    model = xgb.Booster()
    model.load_model("xgboost_temperature_model.json")
    return model

@st.cache_resource
def load_vehicle_model():
    model = xgb.Booster()
    model.load_model("xgboost_vehicle_model.json")
    return model

@st.cache_resource
def load_battery_model():
    model = xgb.Booster()
    model.load_model("xgboost_battery_model.json")
    return model


# ===============================================
# 1️⃣ BV TEMPERATURE PREDICTION
# ===============================================
if app_mode == "BV Temperature Prediction":

    st.header("🔋 BV Temperature Range Prediction System")

    model = load_temperature_model()

    # Load dataset to extract feature names
    data = pd.read_csv("BV_Temperature_Range_Dataset_2000.csv")
    feature_columns = list(data.columns[:-1])

    st.subheader("Enter Input Values")

    user_input = {}
    for col in feature_columns:
        user_input[col] = st.number_input(f"Enter {col}", value=0.0)

    if st.button("Predict Temperature Status"):

        input_df = pd.DataFrame([user_input])
        dmatrix = xgb.DMatrix(input_df)

        prediction_prob = model.predict(dmatrix)[0]
        prediction = 1 if prediction_prob > 0.5 else 0

        if prediction == 0:
            st.success("✅ Status: NORMAL")
        else:
            st.error("⚠️ Status: WARNING")

        st.info(f"Prediction Probability: {round(float(prediction_prob), 4)}")


# ===============================================
# 2️⃣ VEHICLE HEALTH PREDICTION
# ===============================================
elif app_mode == "Vehicle Health Prediction":

    st.header("🚗 Vehicle Health Prediction System")

    model = load_vehicle_model()
    label_encoder = joblib.load("label_encoder.pkl")

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

    if st.button("Predict Vehicle Status"):

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


# ===============================================
# 3️⃣ BATTERY HEALTH PREDICTION
# ===============================================
elif app_mode == "Battery Health Prediction":

    st.header("🔋 Battery Health Prediction System")

    model = load_battery_model()

    # Label Encoder (same as training)
    le = LabelEncoder()
    le.fit(["Low Battery", "Normal Battery"])

    bv = st.number_input(
        "Battery Voltage (BV)",
        min_value=0.0,
        max_value=20.0,
        value=12.0
    )

    temperature = st.number_input(
        "Temperature (°C)",
        min_value=0.0,
        max_value=100.0,
        value=25.0
    )

    gas_flag = st.selectbox("Gas Flag (G/F)", [0, 1])

    battery_percentage = st.number_input(
        "Battery Percentage (BP)",
        min_value=0.0,
        max_value=100.0,
        value=75.0
    )

    if st.button("Predict Battery Status"):

        input_data = pd.DataFrame(
            [[bv, temperature, gas_flag, battery_percentage]],
            columns=["BV", "Temperature", "Gas_Flag", "Battery_Percentage"]
        )

        dmatrix = xgb.DMatrix(input_data)

        prediction_prob = model.predict(dmatrix)[0]
        prediction = 1 if prediction_prob > 0.5 else 0

        status = le.inverse_transform([prediction])[0]

        st.subheader("Prediction Result")

        if status == "Normal Battery":
            st.success(f"✅ Status: {status}")
        else:
            st.error(f"⚠ Status: {status}")

        st.write(f"Prediction Probability: {round(prediction_prob, 4)}")