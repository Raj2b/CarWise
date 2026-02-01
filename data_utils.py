# data_utils.py
import pandas as pd
from sklearn.model_selection import train_test_split

#this file mainly handles:
# 1. loading and cleaning the main user car dataset
# 2. splitting data into train/val/test sets


#5 price bands (our target classes)
def _price_to_band(price: float) -> int:
    if price <= 8000:  #Band 0: $0 - $8,000
        return 0
    if price <= 15000:  #Band 1: $8,001 - $15,000
        return 1
    if price <= 25000:  #Band 2: $15,001 - $25,000
        return 2
    if price <= 40000:  #Band 3: $25,001 - $40,000
        return 3
    return 4            #Band 4: $40,001 and above



#load the raw CSV, do light cleaning and add a descrete price_band label
def load_and_clean_data(csv_path: str = "data.csv", max_rows: int | None = None) -> pd.DataFrame:

    print("Loading data...")
    df = pd.read_csv(csv_path)
    print(f"Raw shape: {df.shape}")
    print("Columns:", list(df.columns))

    #basic cleaning
    df = df.copy()
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["price"])
    df = df[(df["price"] >= 500) & (df["price"] <= 80000)]

    if "description" in df.columns:
        df["description"] = df["description"].fillna("").astype(str)

    df = df.drop_duplicates()

    print(f"After cleaning: {df.shape}")

    #downsample to max_rows for speed ---
    if max_rows is not None and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42)
        df = df.reset_index(drop=True)
        print(f"Subsampled to {len(df)} rows for faster experiments.")

    #price_band label
    df["price_band"] = df["price"].apply(_price_to_band)

    #Show a small preview of price vs band(sanity check)
    preview = df[["price", "price_band"]].head()
    print("With price_band:", preview.shape)
    print(preview)

    #Print how many exampels we have in each band
    print("Class counts (price_band):")
    print(df["price_band"].value_counts().sort_index())

    return df


#Split the cleaned data into train/validation/test sets
def make_splits(
    df: pd.DataFrame,
    target_col: str = "price_band",
    random_state: int = 42,
):
    feature_cols = [  #features we use as input (spec-only + text)
        "year",
        "manufacturer",
        "model",
        "condition",
        "cylinders",
        "fuel",
        "odometer",
        "title_status",
        "transmission",
        "type",
        "description",
    ]

    feature_cols = [c for c in feature_cols if c in df.columns]

    # X = inputs (features), y = labels (price bands)
    X = df[feature_cols].copy()
    y = df[target_col].values

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.25,
        random_state=random_state,
        stratify=y_train_val,
    )

    print(f"Train / Val / Test sizes: {len(y_train)} {len(y_val)} {len(y_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


# This data loader does not require a price column and only lightly cleans teh columms
def load_unlabelled_data(csv_path: str) -> pd.DataFrame:
    # This loads our test CSV
    print(f"\nLoading UNLABELLED data from {csv_path} ...")
    df = pd.read_csv(csv_path)
    print(f"Raw shape (unlabelled): {df.shape}")
    print("Columns:", list(df.columns))

    # Make sure description is a clean string
    if "description" in df.columns:
        df["description"] = df["description"].fillna("").astype(str)

    # Make year / odometer numeric if present (preprocessor will impute missing)
    for col in ["year", "odometer"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df