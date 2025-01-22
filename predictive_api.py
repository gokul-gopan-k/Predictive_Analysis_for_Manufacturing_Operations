from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, recall_score
import joblib
import os
import uvicorn
# Initialize FastAPI app
app = FastAPI()

# Initialize global variables
data = None
model = None
scaler = None

# Define input schema for prediction
class PredictInput(BaseModel):
    Hydraulic_Pressure: float
    Coolant_Pressure: float
    Air_System_Pressure: float
    Coolant_Temperature: float
    Hydraulic_Oil_Temperature: float
    Spindle_Bearing_Temperature: float
    Spindle_Vibration: float
    Tool_Vibration: float
    Spindle_Speed: float
    Voltage: float
    Torque: float
    Cutting: float
    

# Function to filter outliers in a column
def filter_outliers(df, column_name):
    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)]

@app.get('/')
def index():
    return {'message: "Hello'}

@app.post("/upload")
def upload_file(file: UploadFile = File(...)):
    global data

    try:
        # Read the uploaded CSV file
        data = pd.read_csv(file.file)
      
        # Drop unnecessary columns
        data = data.drop(["Date", "Machine_ID", "Assembly_Line_No"], axis=1, errors="ignore")
        
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
        data["Downtime"] = data["Downtime"].replace({
            "No_Machine_Failure": 1,
            "Machine_Failure": 0
        })

        # Validate the presence of required columns
        required_columns = {
            "Hydraulic_Pressure", "Coolant_Pressure", "Air_System_Pressure",
            "Coolant_Temperature", "Hydraulic_Oil_Temperature", "Spindle_Bearing_Temperature",
            "Spindle_Vibration", "Tool_Vibration", "Spindle_Speed",
            "Voltage", "Torque", "Cutting","Downtime"
        }
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

@app.post("/train")
def train_model():
    global data, model, scaler

    if data is None:
        raise HTTPException(status_code=400, detail="No data uploaded. Please upload a dataset first.")

    try:
        # Filter outliers in Hydraulic_Pressure(bar)
        data_filtered = filter_outliers(data, "Hydraulic_Pressure")

        # Define the feature columns (including pressure variables)
        feature_columns = [
            "Hydraulic_Pressure", "Coolant_Pressure", "Air_System_Pressure",
            "Coolant_Temperature", "Hydraulic_Oil_Temperature", "Spindle_Bearing_Temperature",
            "Spindle_Vibration", "Tool_Vibration", "Spindle_Speed",
            "Voltage", "Torque", "Cutting"
        ]

        # Validate required columns
        missing_columns = [col for col in feature_columns if col not in data_filtered.columns]
        if missing_columns:
            raise HTTPException(status_code=400, detail=f"Missing columns in dataset: {missing_columns}")

        # Split data into features and target
        X = data_filtered[feature_columns]
        y = data_filtered["Downtime"]

        # Split dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.23, random_state=42)

        # Standardize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Train the Logistic Regression model
        model = DecisionTreeClassifier()
        model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        # Save the model and scaler
        joblib.dump(model, "model.pkl")
        joblib.dump(scaler, "scaler.pkl")

        return {
            "accuracy": accuracy,
            "f1_score": f1,
            "recall": recall
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to train model: {str(e)}")

@app.post("/predict")
def predict(input_data: PredictInput):
    global model, scaler

    if model is None or scaler is None:
        # Load model and scaler if not already loaded
        if os.path.exists("model.pkl") and os.path.exists("scaler.pkl"):
            model = joblib.load("model.pkl")
            scaler = joblib.load("scaler.pkl")
        else:
            raise HTTPException(status_code=400, detail="No trained model found. Please train a model first.")

    try:
        # Create input feature array
        features = np.array([[
            input_data.Hydraulic_Pressure, input_data.Coolant_Pressure, input_data.Air_System_Pressure,
            input_data.Coolant_Temperature, input_data.Hydraulic_Oil_Temperature,
            input_data.Spindle_Bearing_Temperature, input_data.Spindle_Vibration,
            input_data.Tool_Vibration, input_data.Spindle_Speed,
            input_data.Voltage, input_data.Torque, input_data.Cutting
        ]])

        # Standardize input features
        features_scaled = scaler.transform(features)

        # Make prediction
        prediction = model.predict(features_scaled)
        confidence = max(model.predict_proba(features_scaled)[0])

        return {
            "Downtime": "Yes" if prediction[0] == 0 else "No",
            "Confidence": round(confidence, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == '__main__':
    uvicorn.run(app,host='127.0.0.1', port =8000)
