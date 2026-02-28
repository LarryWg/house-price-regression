import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest
import numpy as np

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.inference.inference import predict


@pytest.fixture(scope="session")
def sample_df():
    """Synthetic feature-engineered-like data for inference testing (no file dependency)."""
    return pd.DataFrame({
        "year": [2020, 2020, 2021, 2021, 2022],
        "month": [1, 6, 1, 6, 1],
        "day": [15, 1, 15, 1, 15],
        "zipcode_freq": [10, 5, 10, 3, 8],
        "city_full_encoded": [250_000.0, 180_000.0, 310_000.0, 200_000.0, 275_000.0],
    })


def test_inference_runs_and_returns_predictions(sample_df):
    """Ensure inference pipeline runs and returns predicted_price column."""
    # Use a mock model so the test does not depend on real artifact or training schema
    mock_model = MagicMock()
    n = len(sample_df)
    mock_model.predict.return_value = np.full(n, 250_000.0)

    with patch("src.inference.inference.load", return_value=mock_model):
        preds_df = predict(sample_df)

    # Check output is not empty
    assert not preds_df.empty

    # Must include prediction column
    assert "predicted_price" in preds_df.columns

    # Predictions should be numeric
    assert pd.api.types.is_numeric_dtype(preds_df["predicted_price"])

    # Should have one prediction per input row
    assert len(preds_df["predicted_price"]) == n

    print("✅ Inference pipeline test passed. Predictions:")
    print(preds_df[["predicted_price"]].head())