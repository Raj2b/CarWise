# features.py
from typing import Tuple
import numpy as np
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
 
# This file turns raw DataFrame columns into numeric feature matrices
# that our models can actually use.

# We build:
#   1. SPEC feature matrix  -> from structured fields (year, make, model, etc.)
#   2. TEXT feature matrix  -> from the free-text 'description' column


def build_spec_feature_matrix(X_train_df, X_val_df, X_test_df):
    #Turn the structured SPEC features (year, make, model, etc.)
    #   into a single numeric matrix using:
    #     1. median imputation for numeric cols
    #     2. constant + one-hot for categorical cols
    # numeric features
    numeric_cols = [c for c in ["year", "odometer"] if c in X_train_df.columns]

    # categorical features (reduced list – removing noise columns)
    cat_cols = [
        "manufacturer",
        "model",
        "condition",
        "cylinders",
        "fuel",
        "title_status",
        "transmission",
        "type",
    ]
    cat_cols = [c for c in cat_cols if c in X_train_df.columns]

    # pipelines
    numeric_transformer = SimpleImputer(strategy="median")

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, cat_cols),
        ]
    )

    # fit on train, transform all splits
    X_train = preprocessor.fit_transform(X_train_df)
    X_val = preprocessor.transform(X_val_df)
    X_test = preprocessor.transform(X_test_df)

    # sanity check: no NaNs
    # (for sparse matrices, .data holds the non-zero entries)
    if sparse.issparse(X_train):
        assert not np.isnan(X_train.data).any()
        assert not np.isnan(X_val.data).any()
        assert not np.isnan(X_test.data).any()
    else:
        assert not np.isnan(X_train).any()
        assert not np.isnan(X_val).any()
        assert not np.isnan(X_test).any()

    return X_train, X_val, X_test, preprocessor


def build_text_feature_matrix(X_train_df, X_val_df, X_test_df):
    
    #TEXT-ONLY features from the listing description.
    #Bag-of-words (CountVectorizer) on the 'description' column only.

    text_col = "description"

    #1. Make sure description column exists in all splits 
    for split_df in [X_train_df, X_val_df, X_test_df]:
        if text_col not in split_df.columns:
            raise ValueError(f"Column '{text_col}' not found for text features")

    #2. Clean and extract raw text for each split
    train_text = X_train_df[text_col].fillna("").astype(str)
    val_text = X_val_df[text_col].fillna("").astype(str)
    test_text = X_test_df[text_col].fillna("").astype(str)

    #3. Set up the bag-of-words vectorizer 
    vectorizer = CountVectorizer(
        max_features=3000,  #limit vocabulary size for speed + memory
        min_df=50,          #ignore words that appear in fewer than 50 listings
        stop_words="english",
    )

    #4. Fit on TRAIN text, then transform VAL and TEST 
    X_train_text = vectorizer.fit_transform(train_text)
    X_val_text = vectorizer.transform(val_text)
    X_test_text = vectorizer.transform(test_text)

    return X_train_text, X_val_text, X_test_text, vectorizer

