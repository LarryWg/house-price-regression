import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/raw")

def load_data(file_name: str = "/Users/larry/house-price-regression/data/raw/house_prices_raw.csv", output_file: Path | str = DATA_DIR) -> pd.DataFrame:
    """
    Load data from a CSV file and save it to a new location.

    Args:
        file_name (str): The name of the input CSV file.
        output_file (str): The name of the output CSV file.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    # Load the data
    df = pd.read_csv(file_name)
    
    # Save the data to a new location
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    eval_cutoff = pd.Timestamp("2020-01-01")
    holdout_cutoff = pd.Timestamp("2022-01-01")
    train = df[df["date"] < eval_cutoff]
    eval_df = df[(df["date"] >= eval_cutoff) & (df["date"] < holdout_cutoff)]
    holdout = df[df["date"] >= holdout_cutoff]

    # Save train, eval, and holdout data to separate files
    output_dir = Path(output_file)
    train.to_csv(output_dir / "train.csv", index=False)
    eval_df.to_csv(output_dir / "eval.csv", index=False)
    holdout.to_csv(output_dir / "holdout.csv", index=False)

    return train, eval_df, holdout

if __name__ == "__main__":
    load_data()