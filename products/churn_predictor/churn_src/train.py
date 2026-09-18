"""
End-to-end training pipeline.

Run:
    python -m src.train

Produces:
    models/lstm_churn_model.keras
    models/feature_scaler.pkl
    models/categorical_encoders.pkl
    models/feature_columns.json
    models/shap_background.npy   (small sample of scaled train sequences, for SHAP later)
    artifacts/training_history.json
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from tensorflow import keras

from churn_src import config
from churn_src.data_cleaning import clean_data, run_eda
from churn_src.feature_engineering import engineer_features
from churn_src.sequence_preparation import build_sequences, customer_level_split
from churn_src.scaling import fit_scaler, transform_sequences, save_scaler
from churn_src.model import build_lstm_model


def load_and_prepare():
    raw = pd.read_csv(config.RAW_DATA_PATH)
    cleaned = clean_data(raw)
    run_eda(cleaned)

    train_ids, val_ids, test_ids = customer_level_split(cleaned)

    train_df = cleaned[cleaned[config.ID_COL].isin(train_ids)]
    val_df = cleaned[cleaned[config.ID_COL].isin(val_ids)]
    test_df = cleaned[cleaned[config.ID_COL].isin(test_ids)]

    # Fit feature engineering / encoders on TRAIN only, then apply everywhere
    train_fe, encoders = engineer_features(train_df, encoders=None)
    val_fe, _ = engineer_features(val_df, encoders=encoders)
    test_fe, _ = engineer_features(test_df, encoders=encoders)

    feature_cols = config.get_full_feature_list()

    X_train, y_train, _ = build_sequences(train_fe, feature_cols)
    X_val, y_val, _ = build_sequences(val_fe, feature_cols)
    X_test, y_test, _ = build_sequences(test_fe, feature_cols)

    scaler = fit_scaler(X_train)
    X_train_s = transform_sequences(X_train, scaler)
    X_val_s = transform_sequences(X_val, scaler)
    X_test_s = transform_sequences(X_test, scaler)

    return {
        "X_train": X_train_s, "y_train": y_train,
        "X_val": X_val_s, "y_val": y_val,
        "X_test": X_test_s, "y_test": y_test,
        "scaler": scaler, "encoders": encoders, "feature_cols": feature_cols,
    }


def train_model(data: dict, epochs: int = None):
    epochs = epochs or config.EPOCHS
    n_features = len(data["feature_cols"])
    model = build_lstm_model(n_features=n_features)

    # Handle class imbalance
    y_train = data["y_train"]
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    class_weight = None
    if n_pos > 0 and n_neg > 0:
        class_weight = {0: len(y_train) / (2 * n_neg), 1: len(y_train) / (2 * n_pos)}

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max", patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
        ),
        keras.callbacks.ModelCheckpoint(
            config.MODEL_PATH, monitor="val_auc", mode="max", save_best_only=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc", mode="max", factor=0.5, patience=4, min_lr=1e-5,
        ),
    ]

    history = model.fit(
        data["X_train"], data["y_train"],
        validation_data=(data["X_val"], data["y_val"]),
        epochs=epochs,
        batch_size=config.BATCH_SIZE,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2,
    )
    return model, history


def save_artifacts(data: dict, history):
    save_scaler(data["scaler"])
    joblib.dump(data["encoders"], config.ENCODERS_PATH)

    with open(config.FEATURE_LIST_PATH, "w") as f:
        json.dump(data["feature_cols"], f, indent=2)

    # Small background sample for SHAP (scaled, already-padded sequences)
    bg_size = min(100, data["X_train"].shape[0])
    idx = np.random.choice(data["X_train"].shape[0], bg_size, replace=False)
    np.save(config.SHAP_BACKGROUND_PATH, data["X_train"][idx])

    hist_path = os.path.join(config.ARTIFACTS_DIR, "training_history.json")
    with open(hist_path, "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f, indent=2)

    print(f"Saved model      -> {config.MODEL_PATH}")
    print(f"Saved scaler     -> {config.SCALER_PATH}")
    print(f"Saved encoders   -> {config.ENCODERS_PATH}")
    print(f"Saved feature list -> {config.FEATURE_LIST_PATH}")
    print(f"Saved SHAP background -> {config.SHAP_BACKGROUND_PATH}")
    print(f"Saved training history -> {hist_path}")


if __name__ == "__main__":
    data = load_and_prepare()
    model, history = train_model(data)
    save_artifacts(data, history)

    # Stash test split for the separate evaluate.py script
    np.savez(
        os.path.join(config.ARTIFACTS_DIR, "test_split.npz"),
        X_test=data["X_test"], y_test=data["y_test"],
    )
    print("Training complete. Run `python -m src.evaluate` for full evaluation + plots.")
