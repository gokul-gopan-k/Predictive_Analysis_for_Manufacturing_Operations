# Predictive_Analysis_for_Manufacturing_Operations

## Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. **Create a Virtual Environment**
   ```bash
   python3 -m venv env
   source env/bin/activate   # For Linux/MacOS
   env\Scripts\activate    # For Windows
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**
   ```bash
   uvicorn main:app --host 127.0.0.1 --port 8000
   ```

## API Endpoint using curl

### 1. Root Endpoint
#### **GET /**
- **Description**: Check if the API is running.
- **Request**:
  ```
  curl -X 'GET' \
  'http://127.0.0.1:8000/' \
  -H 'accept: application/json'
  ```
- **Response**:
  ```json
  {
    "message": "Hello"
  }
  ```

---

### 2. Upload Dataset
#### **POST /upload**
- **Description**: Upload a CSV dataset for model training.
- **Request**:
  ```
  curl -X 'POST' \
  'http://127.0.0.1:8000/upload' \
  -H 'accept: application/json' \
  -F 'file=@<your-dataset.csv>'
  ```
- **Response** (success):
  ```json
  {
    "message": "File uploaded successfully",
    "rows": 3500,
    "columns": ["Hydraulic_Pressure", "Coolant_Pressure", ...]
  }
  ```
- **Response** (failure):
  ```json
  {
    "detail": "CSV must contain columns: {'Downtime', 'Hydraulic_Pressure', ...}"
  }
  ```

---

### 3. Train Model
#### **POST /train**
- **Description**: Train the model using the uploaded dataset.
- **Request**:
  ```
  curl -X 'POST' \
  'http://127.0.0.1:8000/train' \
  -H 'accept: application/json'
  ```
- **Response** (success):
  ```json
  {
    "accuracy": 0.92,
    "f1_score": 0.85,
    "recall": 0.87
  }
  ```
- **Response** (failure):
  ```json
  {
    "detail": "No data uploaded. Please upload a dataset first."
  }
  ```

---

### 4. Make Prediction
#### **POST /predict**
- **Description**: Predict machine downtime using input feature values.
- **Request**:
  ```
  curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "Hydraulic_Pressure": 15.2,
    "Coolant_Pressure": 1.3,
    "Air_System_Pressure": 0.8,
    "Coolant_Temperature": 45.0,
    "Hydraulic_Oil_Temperature": 60.5,
    "Spindle_Bearing_Temperature": 50.2,
    "Spindle_Vibration": 0.12,
    "Tool_Vibration": 0.15,
    "Spindle_Speed": 1500,
    "Voltage": 220,
    "Torque": 50,
    "Cutting": 5.2
  }'
  ```
- **Response**:
  ```json
  {
    "Downtime": "No",
    "Confidence": 0.89
  }
  ```

## Notes
- **Dataset Requirements**:
  The uploaded CSV must contain these columns:
  - `Hydraulic_Pressure`, `Coolant_Pressure`, `Air_System_Pressure`, `Coolant_Temperature`, `Hydraulic_Oil_Temperature`, `Spindle_Bearing_Temperature`, `Spindle_Vibration`, `Tool_Vibration`, `Spindle_Speed`, `Voltage`, `Torque`, `Cutting`, and `Downtime`.

- **Model Outputs**:
  - `Downtime`: Indicates whether machine downtime is predicted (`Yes` or `No`).
  - `Confidence`: Probability of the predicted outcome.

- **Saved Artifacts**:
  Trained model and scaler are saved as `model.pkl` and `scaler.pkl` in the project directory.
