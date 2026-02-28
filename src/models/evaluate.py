from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
from joblib import load
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from typing import Dict, Optional

DEFAULT_EVALUATION = Path("data/processed/eval_engineered.csv")
DEFAULT_MODEL = Path("models/xgb_model.pkl")

def _maybe_sample(df: pd.DataFrame, sample_frac: Optional[float], random_state: int) -> pd.DataFrame:
    if sample_frac is None:
        return df
    sample_frac = float(sample_frac)
    if sample_frac <= 0 or sample_frac >= 1:
        return df
    return df.sample(frac=sample_frac, random_state=random_state).reset_index(drop=True)


def evaluate_model(
    model_path: Path | str = DEFAULT_MODEL,
    eval_path: Path | str = DEFAULT_EVALUATION,
    sample_frac: Optional[float] = None,
    random_state: int = 42,
) -> Dict[str, float]:
    eval_df = pd.read_csv(eval_path)
    eval_df = _maybe_sample(eval_df, sample_frac, random_state)

    target = "price"
    X_eval, y_eval = eval_df.drop(columns=[target]), eval_df[target]

    model = load(model_path)
    y_pred = model.predict(X_eval)

    mae = float(mean_absolute_error(y_eval, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_eval, y_pred)))
    r2 = float(r2_score(y_eval, y_pred))
    metrics = {"mae": mae, "rmse": rmse, "r2": r2}

    print("📊 Evaluation:")
    print(f"   MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.4f}")
    return metrics


if __name__ == "__main__":
    evaluate_model()