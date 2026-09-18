"""
Data cleaning, exploratory data analysis helpers, and the
train/validation/test split + preprocessing pipeline (tabular encoders,
scalers) used by both training and inference.

Run directly to clean the raw CSV, print an EDA summary, and write out
train.csv / val.csv / test.csv plus a fitted preprocessor.joblib.

    python src/data_preprocessing.py
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.preprocessor import TabularPreprocessor


# ----------------------------------------------------------------------
# Cleaning
# ----------------------------------------------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    before = len(df)
    df = df.drop_duplicates(subset=[config.ID_COLUMN])
    print(f"Removed {before - len(df)} duplicate rows.")

    # Impute numeric missing values with median
    for col in config.NUMERIC_FEATURES:
        if col in df.columns and df[col].isna().any():
            median_val = df[col].median()
            n_missing = df[col].isna().sum()
            df[col] = df[col].fillna(median_val)
            print(f"Imputed {n_missing} missing values in '{col}' with median={median_val:.4f}")

    # Fill missing text with empty string
    for col in config.TEXT_COLUMNS:
        df[col] = df[col].fillna("")

    # Drop rows missing critical fields
    critical = [config.IMAGE_COLUMN] + config.TARGET_COLUMNS
    before = len(df)
    df = df.dropna(subset=critical)
    print(f"Dropped {before - len(df)} rows missing critical fields.")

    # Clip obviously invalid values
    df["audience_size"] = df["audience_size"].clip(lower=1)
    df["discount_percent"] = df["discount_percent"].clip(0, 100)
    df["send_hour"] = df["send_hour"].clip(0, 23)
    df["send_day"] = df["send_day"].clip(0, 6)

    for col in ["open_rate", "ctr", "conversion_rate", "unsubscribe_rate"]:
        df[col] = df[col].clip(0, 1)

    df = df.reset_index(drop=True)
    return df


# ----------------------------------------------------------------------
# EDA
# ----------------------------------------------------------------------
def run_eda(df: pd.DataFrame):
    print("\n===== EDA SUMMARY =====")
    print(f"Rows: {len(df)}  Columns: {len(df.columns)}")
    print("\n-- Missing values --")
    print(df.isna().sum()[df.isna().sum() > 0])

    print("\n-- Numeric feature stats --")
    print(df[config.NUMERIC_FEATURES].describe().T)

    print("\n-- Target stats --")
    print(df[config.TARGET_COLUMNS].describe().T)

    print("\n-- Segment distribution --")
    print(df["segment"].value_counts())

    print("\n-- Campaign type distribution --")
    print(df["campaign_type"].value_counts())

    print("\n-- Correlation of numeric features with performance_score --")
    corr = df[config.NUMERIC_FEATURES + ["performance_score"]].corr()["performance_score"].sort_values(ascending=False)
    print(corr)
    print("========================\n")


# ----------------------------------------------------------------------
# Feature engineering
# ----------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Recompute body_length in case body text was edited/cleaned
    df["body_length"] = df["body"].astype(str).apply(len)
    df["subject_length"] = df["subject"].astype(str).apply(len)
    df["links_per_100_words"] = df["num_links"] / (df["body_length"].clip(lower=1) / 100)
    return df


# ----------------------------------------------------------------------
# Split + save
# ----------------------------------------------------------------------
def split_and_save(df: pd.DataFrame):
    train_val_df, test_df = train_test_split(
        df, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED
    )
    train_df, val_df = train_test_split(
        train_val_df, test_size=config.VAL_SIZE, random_state=config.RANDOM_SEED
    )

    train_df.to_csv(config.TRAIN_CSV, index=False)
    val_df.to_csv(config.VAL_CSV, index=False)
    test_df.to_csv(config.TEST_CSV, index=False)

    print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")

    preprocessor = TabularPreprocessor().fit(train_df)
    joblib.dump(preprocessor, config.PREPROCESSOR_PATH)
    print(f"Saved fitted preprocessor to {config.PREPROCESSOR_PATH}")

    return train_df, val_df, test_df, preprocessor


def load_and_prepare():
    df = pd.read_csv(config.RAW_CSV_PATH)
    df = clean_data(df)
    df = engineer_features(df)
    return df


if __name__ == "__main__":
    df = load_and_prepare()
    run_eda(df)
    split_and_save(df)
