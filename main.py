from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from typing import List
import pandas as pd

# Initialize FastAPI app
app = FastAPI(
    title="Iris Species Prediction API",
    description="API for predicting iris species using machine learning",
    version="1.0.0"
)

# Load the trained model and feature names
try:
    model = joblib.load('iris_model.pkl')
    feature_names = joblib.load('feature_names.pkl')
except FileNotFoundError:
    raise Exception("Model files not found. Please run train_model.py first.")

# Pydantic model for input validation
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class PredictionResponse(BaseModel):
    prediction: str
    probability: dict

@app.get("/")
async def root():
    return {"message": "Iris Species Prediction API", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict", response_model=PredictionResponse)
async def predict_species(features: IrisFeatures):
    try:
        # Convert input to DataFrame
        input_data = pd.DataFrame([features.dict()])
        
        # Ensure feature order matches training data
        input_data = input_data[feature_names]
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        
        # Get class names
        class_names = model.classes_
        
        # Create probability dictionary
        prob_dict = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}
        
        return PredictionResponse(
            prediction=prediction,
            probability=prob_dict
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict_batch")
async def predict_batch(features_list: List[IrisFeatures]):
    try:
        predictions = []
        for features in features_list:
            input_data = pd.DataFrame([features.dict()])
            input_data = input_data[feature_names]
            
            prediction = model.predict(input_data)[0]
            probabilities = model.predict_proba(input_data)[0]
            class_names = model.classes_
            prob_dict = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}
            
            predictions.append({
                "prediction": prediction,
                "probability": prob_dict
            })
        
        return {"predictions": predictions}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

@app.get("/model_info")
async def model_info():
    return {
        "model_type": "RandomForestClassifier",
        "features": feature_names,
        "classes": model.classes_.tolist(),
        "n_estimators": model.n_estimators
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
