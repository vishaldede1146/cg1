"""
Data cleaning + exploratory data analysis (EDA).

clean_data(df)      -> returns a cleaned DataFrame (dedup, missing values, outliers, dtypes)
run_eda(df, out_dir) -> saves summary stats + plots to artifacts/ for a quick visual sanity check
"""

import os
import pandas as pd
import numpy as np

from churn_src import config


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Drop exact duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    print(f"[clean_data] Dropped {before - len(df)} duplicate rows")

    # 2. Sort so per-customer sequences are chronological
    df = df.sort_values([config.ID_COL, config.TIME_COL]).reset_index(drop=True)

    # 3. Fix dtypes
    for col in config.NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 4. Impute missing numeric values per-customer (forward/backward fill),
    #    then fall back to global median for any customer missing entirely.
    for col in config.NUMERIC_FEATURES:
        df[col] = df.groupby(config.ID_COL)[col].transform(lambda s: s.ffill().bfill())
        df[col] = df[col].fillna(df[col].median())

    # 5. Clip physically impossible negative values
    non_negative_cols = [
        "LoginCount", "SessionDuration", "PurchaseAmount", "MonthlyCharge",
        "DataUsage", "Complaints", "SupportCalls", "PaymentDelay", "Tenure",
    ]
    for col in non_negative_cols:
        df[col] = df[col].clip(lower=0)

    # 6. Winsorize extreme outliers (1st / 99th percentile) per numeric column
    for col in config.NUMERIC_FEATURES:
        lower, upper = df[col].quantile([0.01, 0.99])
        df[col] = df[col].clip(lower=lower, upper=upper)

    # 7. Clean categorical columns
    for col in config.CATEGORICAL_FEATURES:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].where(df[col].isin(
            config.CONTRACT_TYPES if col == "ContractType" else config.PLAN_TYPES
        ), other=df[col].mode()[0])

    # 8. Ensure target is int 0/1
    df[config.TARGET_COL] = df[config.TARGET_COL].astype(int)

    return df


def run_eda(df: pd.DataFrame, out_dir: str = None):
    """Saves a small set of PNG plots + a text summary. Safe to skip if matplotlib absent."""
    out_dir = out_dir or config.ARTIFACTS_DIR
    os.makedirs(out_dir, exist_ok=True)

    summary_path = os.path.join(out_dir, "eda_summary.txt")
    with open(summary_path, "w") as f:
        f.write("=== Dataset shape ===\n")
        f.write(f"{df.shape}\n\n")
        f.write("=== Missing values ===\n")
        f.write(f"{df.isna().sum()}\n\n")
        f.write("=== Numeric summary ===\n")
        f.write(f"{df[config.NUMERIC_FEATURES].describe().T}\n\n")
        churn_rate = df.groupby(config.ID_COL)[config.TARGET_COL].max().mean()
        f.write(f"Customer-level churn rate: {churn_rate:.3f}\n\n")
        f.write("=== Churn rate by ContractType ===\n")
        f.write(
            f"{df.groupby('ContractType')[config.TARGET_COL].mean()}\n\n"
        )
        f.write("=== Churn rate by PlanType ===\n")
        f.write(f"{df.groupby('PlanType')[config.TARGET_COL].mean()}\n")

    print(f"[run_eda] Text summary saved to {summary_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, axes = plt.subplots(2, 2, figsize=(12, 9))

        sns.countplot(
            x=df.groupby(config.ID_COL)[config.TARGET_COL].max(), ax=axes[0, 0]
        )
        axes[0, 0].set_title("Customer-level Churn Distribution")

        corr = df[config.NUMERIC_FEATURES + [config.TARGET_COL]].corr()
        sns.heatmap(corr, annot=False, cmap="coolwarm", ax=axes[0, 1])
        axes[0, 1].set_title("Feature Correlation Heatmap")

        sns.boxplot(data=df, x=config.TARGET_COL, y="SupportCalls", ax=axes[1, 0])
        axes[1, 0].set_title("Support Calls vs Churn")

        sns.boxplot(data=df, x=config.TARGET_COL, y="LoginCount", ax=axes[1, 1])
        axes[1, 1].set_title("Login Count vs Churn")

        plt.tight_layout()
        plot_path = os.path.join(out_dir, "eda_plots.png")
        plt.savefig(plot_path, dpi=120)
        plt.close(fig)
        print(f"[run_eda] Plots saved to {plot_path}")
    except ImportError:
        print("[run_eda] matplotlib/seaborn not installed, skipping plots.")


if __name__ == "__main__":
    raw = pd.read_csv(config.RAW_DATA_PATH)
    cleaned = clean_data(raw)
    run_eda(cleaned)
    cleaned.to_csv(os.path.join(config.DATA_DIR, "cleaned_data.csv"), index=False)
    print("Cleaned data saved to data/cleaned_data.csv")
