# ===============================================
# STREAMLIT APP FOR BV TEMPERATURE MODEL
# ===============================================

import streamlit as st
import pandas as pd
import xgboost as xgb

# -----------------------------------------------
# PAGE CONFIG
# -----------------------------------------------
st.set_page_config(page_title="BV Temperature Prediction", layout="centered")

st.title("🔋 BV Temperature Range Prediction System")

# -----------------------------------------------
# LOAD TRAINED MODEL
# -----------------------------------------------
@st.cache_resource
def load_model():
    model = xgb.Booster()
    model.load_model("xgboost_temperature_model.json")
    return model

model = load_model()

# -----------------------------------------------
# LOAD DATASET TO GET FEATURE NAMES
# -----------------------------------------------
data = pd.read_csv("BV_Temperature_Range_Dataset_2000.csv")
feature_columns = list(data.columns[:-1])   # All except label

st.subheader("Enter Input Values")

# -----------------------------------------------
# CREATE INPUT FIELDS
# -----------------------------------------------
user_input = {}

for col in feature_columns:
    user_input[col] = st.number_input(f"Enter {col}", value=0.0)

# -----------------------------------------------
# PREDICTION BUTTON
# -----------------------------------------------
if st.button("Predict"):

    input_df = pd.DataFrame([user_input])
    dmatrix = xgb.DMatrix(input_df)

    prediction_prob = model.predict(dmatrix)[0]
    prediction = 1 if prediction_prob > 0.5 else 0

    # -------------------------------------------
    # CONDITION MAPPING
    # -------------------------------------------
    if prediction == 0:
        st.success("✅ Status: NORMAL")
    else:
        st.error("⚠️ Status: WARNING")

    st.info(f"Prediction Probability: {round(float(prediction_prob), 4)}")