# Iris Species Prediction API

A FastAPI application for predicting iris species using machine learning.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Train the model:
```bash
python train_model.py
```

3. Start the API server:
```bash
python main.py
```

The API will be available at `http://localhost:8000`

## API Endpoints

- `GET /` - Root endpoint with basic info
- `GET /health` - Health check
- `POST /predict` - Predict single iris species
- `POST /predict_batch` - Predict multiple iris species
- `GET /model_info` - Get model information

## Example Usage

```python
import requests

# Single prediction
data = {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
}

response = requests.post("http://localhost:8000/predict", json=data)
print(response.json())
```

## Documentation

Visit `http://localhost:8000/docs` for interactive API documentation.
