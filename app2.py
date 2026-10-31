import streamlit as st
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

# -------------------------
# Page Config
# -------------------------
st.set_page_config(page_title="Battery Health Predictor", layout="centered")

st.title("🔋 Battery Health Prediction System")
st.write("Enter battery parameters to predict status")

# -------------------------
# Load Model
# -------------------------
model = xgb.Booster()
model.load_model("xgboost_battery_model.json")

# Label Encoder (same as training)
le = LabelEncoder()
le.fit(["Low Battery", "Normal Battery"])

# -------------------------
# User Input
# -------------------------
bv = st.number_input("Battery Voltage (BV)", min_value=0.0, max_value=20.0, value=12.0)
temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=100.0, value=25.0)
gas_flag = st.selectbox("Gas Flag (G/F)", [0, 1])
battery_percentage = st.number_input("Battery Percentage (BP)", min_value=0.0, max_value=100.0, value=75.0)

# -------------------------
# Prediction
# -------------------------
if st.button("Predict Battery Status"):

    input_data = pd.DataFrame([[bv, temperature, gas_flag, battery_percentage]],
                              columns=["BV", "Temperature", "Gas_Flag", "Battery_Percentage"])

    dmatrix = xgb.DMatrix(input_data)

    prediction_prob = model.predict(dmatrix)[0]
    prediction = 1 if prediction_prob > 0.5 else 0

    status = le.inverse_transform([prediction])[0]

    st.subheader("Prediction Result:")
    
    if status == "Normal Battery":
        st.success(f"✅ Status: {status}")
    else:
        st.error(f"⚠ Status: {status}")

    st.write(f"Prediction Probability: {round(prediction_prob, 4)}")