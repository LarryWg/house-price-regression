import pandas as pd
from pathlib import Path
import re 

RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

CITY_MAPPING = {
    'Las Vegas-Henderson-Paradise': 'Las Vegas-Henderson-North Las Vegas',
    'Denver-Aurora-Lakewood': 'Denver-Aurora-Centennial',
    'Houston-The Woodlands-Sugar Land': 'Houston-Pasadena-The Woodlands',
    'Austin-Round Rock-Georgetown': 'Austin-Round Rock-San Marcos',
    'Miami-Fort Lauderdale-Pompano Beach': 'Miami-Fort Lauderdale-West Palm Beach',
    'San Francisco-Oakland-Berkeley': 'San Francisco-Oakland-Fremont',
    'DC_Metro': 'Washington-Arlington-Alexandria',
    'Atlanta-Sandy Springs-Alpharetta': 'Atlanta-Sandy Springs-Roswell'
}


def normalize_city_names(s: str) -> str:
    """
    Normalize city names by replacing specific patterns.

    Args:
        city_name (str): The original city name.

    Returns:
        str: The normalized city name.
    """
    if pd.isna(s):
        return s
    
    s = str(s).strip()

    s = re.sub(r"[–—-]", "-", s)          # unify dashes
    s = re.sub(r"\s+", " ", s)   
    return s


    
def clean_merge(df: pd.DataFrame, metros_path: str | None = "data/raw/usmetros.csv") -> pd.DataFrame:
    """Clean and merge the input DataFrame with metro area data."""
    if "city_full" not in df.columns:
        print("Input DataFrame must contain 'city_full' column.")
        return df
    
    df["city_full"] = df["city_full"].apply(normalize_city_names)
    normalize_mapping = {normalize_city_names(k): v for k, v in CITY_MAPPING.items()}
    df["city_full"] = df["city_full"].replace(normalize_mapping)

    if not metros_path or not Path(metros_path).exists():
        print("Skipping lat/lng merge: metros file not provided or not found.")
        return df

    # Merge lat/lng
    metros = pd.read_csv(metros_path)
    if "metro_full" not in metros.columns or not {"lat", "lng"}.issubset(metros.columns):
        print("Skipping lat/lng merge: metros file missing required columns.")
        return df

    metros["metro_full"] = metros["metro_full"].apply(normalize_city_names)
    df = df.merge(metros[["metro_full", "lat", "lng"]],
                  how="left", left_on="city_full", right_on="metro_full")
    df.drop(columns=["metro_full"], inplace=True, errors="ignore")

    missing = df[df["lat"].isnull()]["city_full"].unique()
    if len(missing) > 0:
        print("Still missing lat/lng for:", missing)
    else:
        print("All cities matched with metros dataset.")
    return df




def duplicated_data(df: pd.DataFrame) -> pd.DataFrame:
    """Identify duplicated rows in the DataFrame."""
    df = df.drop_duplicates(subset=df.columns.difference(["date", "year"]), keep=False)
    print("Removed duplicated rows based on all columns except 'date' and 'year'.")
    return df

def outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Identify and handle outliers in the DataFrame."""
    if "median_listing_price" not in df.columns:
        return df
    
    df = df[df["median_listing_price"] < 19_000_000].copy()
    return df

def preprocess(split: str, raw_dir: Path | str = RAW_DATA_DIR, processed_dir: Path | str = PROCESSED_DATA_DIR, metros_path: str | None = "data/raw/usmetros.csv") -> pd.DataFrame:
    """Preprocess the data for a given split."""
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    path = raw_dir / f"{split}.csv"
    df = pd.read_csv(path)

    df = clean_merge(df, metros_path=metros_path)
    df = duplicated_data(df)
    df = outliers(df)

    out_path = processed_dir / f"cleaning_{split}.csv"
    df.to_csv(out_path, index=False)
    return df
    

def preprocess_run( splits: tuple[str, ...] = ("train", "eval", "holdout"),
    raw_dir: Path | str = RAW_DATA_DIR,
    processed_dir: Path | str = PROCESSED_DATA_DIR,
    metros_path: str | None = "data/raw/usmetros.csv"):
    for split in splits:
        preprocess(split, raw_dir=raw_dir, processed_dir=processed_dir, metros_path=metros_path)


if __name__ == "__main__":
    preprocess_run()