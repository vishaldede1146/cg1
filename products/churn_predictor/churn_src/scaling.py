"""
Scaling for 3D sequence tensors (n_samples, timesteps, n_features).

We fit a StandardScaler on the flattened (n_samples*timesteps, n_features)
TRAIN data only (real rows; padded zero-rows are harmless to include since
they are masked out by the model, but to keep the scaler honest we exclude
fully-zero padded rows from fitting).
"""

import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

from churn_src import config


def fit_scaler(X_train: np.ndarray) -> StandardScaler:
    n, t, f = X_train.shape
    flat = X_train.reshape(-1, f)
    # Exclude padded (all-zero) rows from fitting statistics
    non_padded_mask = ~np.all(flat == 0, axis=1)
    scaler = StandardScaler()
    scaler.fit(flat[non_padded_mask])
    return scaler


def transform_sequences(X: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    n, t, f = X.shape
    flat = X.reshape(-1, f)
    non_padded_mask = ~np.all(flat == 0, axis=1)

    flat_scaled = flat.copy()
    flat_scaled[non_padded_mask] = scaler.transform(flat[non_padded_mask])
    # padded rows remain exactly 0, which the Masking layer will ignore
    return flat_scaled.reshape(n, t, f)


def save_scaler(scaler: StandardScaler, path: str = None):
    path = path or config.SCALER_PATH
    joblib.dump(scaler, path)


def load_scaler(path: str = None) -> StandardScaler:
    path = path or config.SCALER_PATH
    return joblib.load(path)
