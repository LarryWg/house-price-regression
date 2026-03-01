from fastapi import FastAPI
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import boto3, os #AWS SDK for Python

from src.inference.inference import predict


# configs
S3_BUCKET = os.getenv("S3_BUCKET", "larry-house-price-regression-data")
REGION = os.getenv("AWS_REGION", "ca-central-1")
s3 = boto3.client("s3", region_name=REGION)

def load_data_from_s3(key, local_path):
    """Download from s3 if not cached locally."""

    local_path = Path(local_path)
    if not local_path.exists():
        os.makedirs(local_path.parent, exist_ok=True)
        s3.download_file(S3_BUCKET, key, str(local_path))
        print(f"📥 Downloaded {key} from S3 to {local_path}")

    return str(local_path)


MODEL_PATH = Path(load_data_from_s3("models/xgb_best_model.pkl", "models/xgb_best_model.pkl"))
TRAIN_FE_PATH = Path(load_data_from_s3("processed/train_engineered.csv", "data/processed/train_engineered.csv"))

if TRAIN_FE_PATH.exists():
    _train_cols = pd.read_csv(TRAIN_FE_PATH, nrows=1)
    TRAIN_FEATURE_COLUMNS = [c for c in _train_cols.columns if c != "price"]  # excluding price column
else:    
    TRAIN_FEATURE_COLUMNS = None


##### APP #####
app = FastAPI(title="House Price Regression API", version="1.0")

# check if api is alive 
@app.get("/")
def root():
    return {"message": "Welcome to the House Price Regression API!"}

# Healh : chcek if model exists and can be loaded
@app.get("/health")
def health_check():
    status: Dict[str, Any] = {"model_path": str(MODEL_PATH)}
    if not MODEL_PATH.exists():
        status["status"] = "error"
        status["message"] = f"Model file not found at {MODEL_PATH}"
    else:
        status["status"] = "healthy"
        if TRAIN_FEATURE_COLUMNS:
            status["n_features"] = len(TRAIN_FEATURE_COLUMNS)

    return status


# Prediction endpoint
@app.post("/predict")
def predict_endpoint(data: List[Dict[str, Any]]):
    if not MODEL_PATH.exists():
        return {"error": f"Model file not found at {MODEL_PATH}"}
    
    df = pd.DataFrame(data)
    if df.empty:
        return {"error": "No data provided for prediction."}

    predictions_df = predict(df, model_path = MODEL_PATH)

    resp = {
        "predictions": predictions_df["predicted_price"].astype(float).tolist()
    }

    if "actual_price" in predictions_df.columns:
        resp["actual_price"] = predictions_df["actual_price"].astype(float).tolist()

    return resp


@app.get("/latest-predictions")
def latest_predictions(limit: int = 5):
    """Endpoint to retrieve the latest predictions from the output CSV."""
    prediction_dir = Path("data/predictions")
    files = sorted(prediction_dir.glob("predictions_*.csv"))
    if not files:
        return {"error": "No prediction files found."}
    latest_file = files[-1]
    df = pd.read_csv(latest_file)
    return {
        "file": latest_file.name,
        "rows": int(len(df)),
        "preview": df.head(limit).to_dict(orient="records")
    }