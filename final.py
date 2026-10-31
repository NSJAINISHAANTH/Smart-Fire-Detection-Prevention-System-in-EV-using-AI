# ============================================
# FINAL FLASK + ARDUINO + 3 AI MODELS
# ============================================

from flask import Flask, render_template
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.preprocessing import LabelEncoder
import serial
import serial.tools.list_ports
import threading
import time
from datetime import datetime

app = Flask(__name__)

# ============================================
# LOAD MODELS
# ============================================

temperature_model = xgb.Booster()
temperature_model.load_model("xgboost_temperature_model.json")

vehicle_model = xgb.Booster()
vehicle_model.load_model("xgboost_vehicle_model.json")

battery_model = xgb.Booster()
battery_model.load_model("xgboost_battery_model.json")

# ============================================
# LOAD FEATURE NAMES FROM TRAINING DATA
# ============================================

# Temperature model features
temp_train_data = pd.read_csv("BV_Temperature_Range_Dataset_2000.csv")
temp_feature_names = list(temp_train_data.columns[:-1])  # All except last (Status)

# Battery model features
battery_train_data = pd.read_csv("battery_dataset.csv")
battery_feature_names = list(battery_train_data.columns[:-1])  # All except last (Status)

# Vehicle model features
vehicle_train_data = pd.read_csv("vehicle_status_dataset_2000.csv")
vehicle_feature_names = list(vehicle_train_data.columns[:-1])  # All except last (Status)

print(f"📊 Temperature features: {temp_feature_names}")
print(f"📊 Battery features: {battery_feature_names}")
print(f"📊 Vehicle features: {vehicle_feature_names}")

label_encoder = joblib.load("label_encoder.pkl")

battery_le = LabelEncoder()
battery_le.fit(["Low Battery", "Normal Battery"])

# ============================================
# SENSOR DATA STORAGE
# ============================================

sensor_data = {
    "temperature": 0.0,
    "temp_status": 0,
    "voltage": 0.0,
    "battery_percentage": 0.0,
    "fire": 0
}

# ============================================
# AUTO DETECT ARDUINO / USE COM9
# ============================================

def find_arduino():
    # Try COM9 first
    try:
        test_serial = serial.Serial("COM7", 9600, timeout=1)
        test_serial.close()
        return "COM7"
    except:
        pass
    
    # Fallback: auto-detect
    ports = serial.tools.list_ports.comports()
    for port in ports:
        if "Arduino" in port.description or "CH340" in port.description:
            return port.device
    return None

arduino = None
port = find_arduino()

if port:
    try:
        arduino = serial.Serial(port, 9600, timeout=1)
        time.sleep(2)
        print(f"✅ Connected to {port}")
    except Exception as e:
        print(f"❌ Serial Error: {e}")
        arduino = None
else:
    print("⚠️ Arduino/COM9 not detected. Using default values.")

# ============================================
# SERIAL READING THREAD (IMPROVED)
# ============================================

def read_serial():
    global sensor_data
    
    while True:
        try:
            if arduino and arduino.is_open:
                if arduino.in_waiting > 0:
                    # Read the first line (label)
                    label = arduino.readline().decode().strip()
                    
                    if label and arduino.in_waiting > 0:
                        # Read the second line (value)
                        value = arduino.readline().decode().strip()
                        
                        if value:
                            try:
                                if label == "T":
                                    sensor_data["temperature"] = float(value)
                                elif label == "S":
                                    sensor_data["temp_status"] = int(value)
                                elif label == "a":
                                    sensor_data["voltage"] = float(value)
                                elif label == "b":
                                    sensor_data["battery_percentage"] = float(value)
                                elif label == "d":
                                    sensor_data["fire"] = int(value)
                                print(f"✅ Read {label}: {value}")
                            except ValueError:
                                print(f"⚠️ Invalid value for {label}: {value}")
        except Exception as e:
            print(f"❌ Serial read error: {e}")
        
        time.sleep(0.1)

# Start thread
if arduino:
    thread = threading.Thread(target=read_serial)
    thread.daemon = True
    thread.start()

# ============================================
# DASHBOARD ROUTE
# ============================================

@app.route("/")
def home():

    try:
        # Use default values if sensor data is still 0
        voltage = sensor_data["voltage"] if sensor_data["voltage"] != 0 else 12.0
        temperature = sensor_data["temperature"] if sensor_data["temperature"] != 0 else 25.0
        fire = sensor_data["fire"]
        battery_percentage = sensor_data["battery_percentage"] if sensor_data["battery_percentage"] != 0 else 75.0

        # ===============================
        # 1️⃣ Temperature Prediction
        # ===============================
        temp_df = pd.DataFrame(
            [[voltage, temperature]],
            columns=temp_feature_names
        )
        dmatrix_temp = xgb.DMatrix(temp_df)
        temp_prob = float(temperature_model.predict(dmatrix_temp)[0])
        temp_pred = 1 if temp_prob > 0.5 else 0
        temp_status_model = "WARNING" if temp_pred == 1 else "NORMAL"

        # ===============================
        # 2️⃣ Battery Prediction
        # ===============================
        battery_df = pd.DataFrame(
            [[voltage, temperature, fire, battery_percentage]],
            columns=battery_feature_names
        )
        dmatrix_battery = xgb.DMatrix(battery_df)
        battery_prob = float(battery_model.predict(dmatrix_battery)[0])
        battery_pred = 1 if battery_prob > 0.5 else 0
        battery_status = battery_le.inverse_transform([battery_pred])[0]

        # ===============================
        # 3️⃣ Vehicle Prediction
        # ===============================
        vehicle_df = pd.DataFrame(
            [[voltage, temperature, fire]],
            columns=vehicle_feature_names
        )
        dmatrix_vehicle = xgb.DMatrix(vehicle_df)
        vehicle_prob = float(vehicle_model.predict(dmatrix_vehicle)[0])
        vehicle_pred = 1 if vehicle_prob > 0.5 else 0
        vehicle_status = label_encoder.inverse_transform([vehicle_pred])[0]

        return render_template(
            "dashboard.html",
            data=sensor_data,
            temp_status_model=temp_status_model,
            temp_prob=round(temp_prob, 4),
            battery_status=battery_status,
            battery_prob=round(battery_prob, 4),
            vehicle_status=vehicle_status,
            vehicle_prob=round(vehicle_prob, 4),
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
    
    except Exception as e:
        import traceback
        print(f"❌ Error in prediction: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        print(f"Sensor data: {sensor_data}")
        
        return render_template(
            "dashboard.html",
            data=sensor_data,
            temp_status_model="NORMAL",
            temp_prob=0.5,
            battery_status="Normal Battery",
            battery_prob=0.5,
            vehicle_status="Vehicle Normal",
            vehicle_prob=0.5,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )


# ============================================
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)