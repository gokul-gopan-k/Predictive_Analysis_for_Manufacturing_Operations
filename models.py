import joblib
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score
from config import MODEL_PATH, SCALER_PATH
from preprocessing import filter_outliers
from fastapi import HTTPException


data= None
model = None
scaler = None

def upload_data(file):
    global data
    try:
        data = pd.read_csv(file.file)

        # Drop unnecessary columns
        data = data.drop(["Date", "Machine_ID", "Assembly_Line_No"], axis=1, errors="ignore")

        # Rename columns
        data = data.rename(columns={
            "Hydraulic_Oil_Temperature(?C)": "Hydraulic_Oil_Temperature",
            "Spindle_Bearing_Temperature(?C)": "Spindle_Bearing_Temperature",
            "Coolant_Temperature(?C)": "Coolant_Temperature",
            "Spindle_Vibration(?m)": "Spindle_Vibration",
            "Tool_Vibration(?m)": "Tool_Vibration",
            "Spindle_Speed(RPM)": "Spindle_Speed",
            "Voltage(volts)": "Voltage",
            "Torque(Nm)": "Torque",
            "Cutting(kN)": "Cutting",
            "Hydraulic_Pressure(bar)": "Hydraulic_Pressure",
            "Coolant_Pressure(bar)": "Coolant_Pressure",
            "Air_System_Pressure(bar)": "Air_System_Pressure"
        })

        # Replace values in the "Downtime" column
        data["Downtime"] = data["Downtime"].replace({"No_Machine_Failure": 1, "Machine_Failure": 0})

        # Validate the presence of required columns
        required_columns = {"Hydraulic_Pressure", "Coolant_Pressure", "Air_System_Pressure",
                            "Coolant_Temperature", "Hydraulic_Oil_Temperature", "Spindle_Bearing_Temperature",
                            "Spindle_Vibration", "Tool_Vibration", "Spindle_Speed", "Voltage", "Torque", "Cutting", "Downtime"}
        missing_columns = required_columns - set(data.columns)
        if missing_columns:
            raise HTTPException(status_code=400, detail=f"CSV must contain columns: {missing_columns}")

        # Fill missing values with column means
        for col in required_columns:
            if data[col].isnull().any():
                data[col].fillna(data[col].mean(), inplace=True)

        return {"message": "File uploaded successfully", "rows": len(data), "columns": list(data.columns)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")

def train_model():
    global data, model, scaler
    try:
        data_filtered = filter_outliers(data, "Hydraulic_Pressure")
        feature_columns = ["Hydraulic_Pressure", "Coolant_Pressure", "Air_System_Pressure", "Coolant_Temperature", 
                           "Hydraulic_Oil_Temperature", "Spindle_Bearing_Temperature", "Spindle_Vibration", "Tool_Vibration", 
                           "Spindle_Speed", "Voltage", "Torque", "Cutting"]
        X = data_filtered[feature_columns]
        y = data_filtered["Downtime"]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.23, random_state=42)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = DecisionTreeClassifier()
        model.fit(X_train, y_train)

        joblib.dump(model, MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)

        # Model evaluation
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        return {"accuracy": round(accuracy,2), "f1_score": round(f1,2) , "recall": round(recall,2)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to train model: {str(e)}")

def predict(input_data):
    global model, scaler
    if model is None or scaler is None:
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
            model = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
        else:
            raise HTTPException(status_code=400, detail="Model not trained.")
    try:
        # Preprocess input and predict
        features = np.array([[
            input_data.Hydraulic_Pressure, input_data.Coolant_Pressure, input_data.Air_System_Pressure,
            input_data.Coolant_Temperature, input_data.Hydraulic_Oil_Temperature,
            input_data.Spindle_Bearing_Temperature, input_data.Spindle_Vibration,
            input_data.Tool_Vibration, input_data.Spindle_Speed,
            input_data.Voltage, input_data.Torque, input_data.Cutting
        ]])
        
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)
        confidence = max(model.predict_proba(features_scaled)[0])

        return {"Downtime": "Yes" if prediction[0] == 0 else "No", "Confidence": round(confidence, 2)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
