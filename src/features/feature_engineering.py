from pathlib import Path
import pandas as pd
from category_encoders import TargetEncoder
from joblib import dump
from pyparsing import col
from pyparsing import col

PROCESSED_DATA_DIR = Path("data/processed")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract date features from the 'date' column."""    
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df.insert(1, "year", df.pop("year"))
    df.insert(2, "month", df.pop("month"))
    df.insert(3, "day", df.pop("day"))
    return df


def frequency_encoding(train: pd.DataFrame, eval: pd.DataFrame, column: str):
    """Apply frequency encoding to a specified column."""
    freq_map = train[column].value_counts()
    train[f"{column}_freq"] = train[column].map(freq_map)
    eval[f"{column}_freq"] = eval[column].map(freq_map)
    return train, eval, freq_map

def target_encoding(train: pd.DataFrame, eval: pd.DataFrame, column: str, target: str):
    """Apply target encoding to a specified column."""
    te = TargetEncoder(cols=[column])
    encoded_col = f"{column}_encoded" if column != "city_full" else "city_full_encoded"
    train[encoded_col] = te.fit_transform(train[column], train[target])
    eval[encoded_col] = te.transform(eval[column])
    return train, eval, te

def drop_unused_columns(train: pd.DataFrame, eval: pd.DataFrame):
    drop_cols = ["date", "city_full", "city", "zipcode", "median_sale_price"]
    train = train.drop(columns=[c for c in drop_cols if c in train.columns], errors="ignore")
    eval = eval.drop(columns=[c for c in drop_cols if c in eval.columns], errors="ignore")
    return train, eval

def run_feature_engineering(
    in_train_path: Path | str | None = None,
    in_eval_path: Path | str | None = None,
    in_holdout_path: Path | str | None = None,
    output_dir: Path | str = PROCESSED_DATA_DIR,
):
    """
    Run feature engineering and write outputs + encoders to disk.
    Applies the same transformations to train, eval, and holdout.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Defaults for inputs
    if in_train_path is None:
        in_train_path = PROCESSED_DATA_DIR / "cleaning_train.csv"
    if in_eval_path is None:
        in_eval_path = PROCESSED_DATA_DIR / "cleaning_eval.csv"
    if in_holdout_path is None:
        in_holdout_path = PROCESSED_DATA_DIR / "cleaning_holdout.csv"

    train_df = pd.read_csv(in_train_path)
    eval_df = pd.read_csv(in_eval_path)
    holdout_df = pd.read_csv(in_holdout_path)

    print("Train date range:", train_df["date"].min(), "to", train_df["date"].max())
    print("Eval date range:", eval_df["date"].min(), "to", eval_df["date"].max())
    print("Holdout date range:", holdout_df["date"].min(), "to", holdout_df["date"].max())

    # Date features
    train_df = date_features(train_df)
    eval_df = date_features(eval_df)
    holdout_df = date_features(holdout_df)

    # Frequency encode zipcode (fit on train only)
    freq_map = None
    if "zipcode" in train_df.columns:
        train_df, eval_df, freq_map = frequency_encoding(train_df, eval_df, "zipcode")
        holdout_df["zipcode_freq"] = holdout_df["zipcode"].map(freq_map).fillna(0)
        dump(freq_map, MODELS_DIR / "freq_encoder.pkl")   # save mapping

    # Target encode city_full (fit on train only)
    target_encoder = None
    if "city_full" in train_df.columns:
        train_df, eval_df, target_encoder = target_encoding(train_df, eval_df, "city_full", "price")
        holdout_df["city_full_encoded"] = target_encoder.transform(holdout_df["city_full"])
        dump(target_encoder, MODELS_DIR / "target_encoder.pkl")  # save encoder

    # Drop leakage / raw categoricals
    train_df, eval_df = drop_unused_columns(train_df, eval_df)
    holdout_df, _ = drop_unused_columns(holdout_df.copy(), holdout_df.copy())

    # Save engineered data
    out_train_path = output_dir / "feature_engineered_train.csv"
    out_eval_path = output_dir / "feature_engineered_eval.csv"
    out_holdout_path = output_dir / "feature_engineered_holdout.csv"
    train_df.to_csv(out_train_path, index=False)
    eval_df.to_csv(out_eval_path, index=False)
    holdout_df.to_csv(out_holdout_path, index=False)

    print("✅ Feature engineering complete.")
    print("   Train shape:", train_df.shape)
    print("   Eval  shape:", eval_df.shape)
    print("   Holdout shape:", holdout_df.shape)
    print("   Encoders saved to models/")

    return train_df, eval_df, holdout_df, freq_map, target_encoder


if __name__ == "__main__":
    run_feature_engineering()
