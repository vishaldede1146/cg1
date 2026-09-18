"""
Feature engineering.

Adds trend/ratio/rolling features that give the LSTM richer signal than raw
monthly snapshots, and encodes categorical columns.

IMPORTANT: fit_encoders() must be called once on training data; the same
encoder mapping is then reused (via encoders dict) for validation/test/inference
so category -> integer codes stay consistent.
"""

import numpy as np
import pandas as pd

from churn_src import config


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values([config.ID_COL, config.TIME_COL])

    grp = df.groupby(config.ID_COL)

    # Month-over-month deltas (0 for each customer's first observed month)
    for col in ["LoginCount", "SessionDuration", "DataUsage", "PurchaseAmount"]:
        df[f"{col}_delta"] = grp[col].diff().fillna(0)

    # 3-month rolling averages of friction signals (min_periods=1 so early months still populate)
    for col in ["Complaints", "SupportCalls", "PaymentDelay"]:
        df[f"{col}_roll3"] = (
            grp[col].transform(lambda s: s.rolling(window=3, min_periods=1).mean())
        )

    # Ratios / derived business metrics
    df["AvgSessionPerLogin"] = df["SessionDuration"] / df["LoginCount"].replace(0, np.nan)
    df["AvgSessionPerLogin"] = df["AvgSessionPerLogin"].fillna(0)

    df["ChargePerGB"] = df["MonthlyCharge"] / df["DataUsage"].replace(0, np.nan)
    df["ChargePerGB"] = df["ChargePerGB"].fillna(df["MonthlyCharge"])

    # Simple composite engagement score (higher = more engaged / healthier)
    df["EngagementScore"] = (
        0.4 * _z(df["LoginCount"])
        + 0.3 * _z(df["SessionDuration"])
        + 0.2 * _z(df["DataUsage"])
        - 0.3 * _z(df["Complaints"])
        - 0.3 * _z(df["SupportCalls"])
        - 0.2 * _z(df["PaymentDelay"])
    )

    return df


def _z(series: pd.Series) -> pd.Series:
    std = series.std()
    if std == 0 or np.isnan(std):
        return series * 0
    return (series - series.mean()) / std


def fit_encoders(df: pd.DataFrame) -> dict:
    """Fit simple category->int mappings on the full category vocabulary."""
    encoders = {}
    for col, vocab in [("ContractType", config.CONTRACT_TYPES), ("PlanType", config.PLAN_TYPES)]:
        encoders[col] = {cat: i for i, cat in enumerate(vocab)}
    return encoders


def apply_encoders(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    df = df.copy()
    for col, mapping in encoders.items():
        default = len(mapping)  # unseen category bucket
        df[f"{col}_encoded"] = df[col].map(mapping).fillna(default).astype(int)
    return df


def engineer_features(df: pd.DataFrame, encoders: dict = None):
    """Full feature engineering step. If encoders is None, fits new ones (training time)."""
    df = add_engineered_features(df)
    if encoders is None:
        encoders = fit_encoders(df)
    df = apply_encoders(df, encoders)
    return df, encoders


if __name__ == "__main__":
    import os
    df = pd.read_csv(os.path.join(config.DATA_DIR, "cleaned_data.csv"))
    df_fe, encoders = engineer_features(df)
    df_fe.to_csv(os.path.join(config.DATA_DIR, "featured_data.csv"), index=False)
    print("Feature-engineered data saved to data/featured_data.csv")
    print("Full feature list:", config.get_full_feature_list())
