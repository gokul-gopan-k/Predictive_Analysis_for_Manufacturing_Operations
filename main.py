from fastapi import FastAPI, HTTPException, File, UploadFile
from .models import upload_data, train_model, predict
from .schemas import PredictInput

app = FastAPI()

@app.get('/')
def index():
    return {"message": "Hello"}

@app.post("/upload")
def upload_file(file: UploadFile = File(...)):
    return upload_data(file)

@app.post("/train")
def train():
    return train_model()

@app.post("/predict")
def predict(input_data: PredictInput):
    return predict(input_data)
