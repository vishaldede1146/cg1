"""
Sequence preparation for the LSTM.

Each customer becomes ONE sample: a (MAX_SEQ_LEN, num_features) matrix built
from their chronological monthly rows, RIGHT-padded with zeros if they have
fewer than MAX_SEQ_LEN months, truncated (most recent months kept) if they
have more. A Masking layer in the model ignores the zero-padded timesteps.

Right-padding (real data first, zeros after) is required rather than
left-padding: TensorFlow's cuDNN-accelerated LSTM kernel only supports
masks of the form [True, True, ..., False, False] (contiguous real steps
on the left, padding on the right). Left-padding works fine on CPU but
throws `InvalidArgumentError` on GPU. Keeping months in chronological
order (oldest first, real data first) also matches how a person would
naturally read the sequence.

Splitting is done at the CUSTOMER level (not row level) to avoid leaking a
customer's other months across train/val/test.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from churn_src import config


def build_sequences(df: pd.DataFrame, feature_cols: list, max_len: int = None):
    """
    Returns:
        X: np.ndarray shape (n_customers, max_len, n_features)
        y: np.ndarray shape (n_customers,)
        customer_ids: list of CustomerID in the same order as X/y
    """
    max_len = max_len or config.MAX_SEQ_LEN
    X, y, ids = [], [], []

    for cust_id, g in df.groupby(config.ID_COL):
        g = g.sort_values(config.TIME_COL)
        feats = g[feature_cols].values.astype(np.float32)
        label = int(g[config.TARGET_COL].iloc[-1])  # churn outcome for this customer

        n = feats.shape[0]
        if n >= max_len:
            seq = feats[-max_len:]  # keep most recent months
        else:
            pad = np.zeros((max_len - n, feats.shape[1]), dtype=np.float32)
            seq = np.vstack([feats, pad])  # right-pad (real data first, zeros after)

        X.append(seq)
        y.append(label)
        ids.append(cust_id)

    return np.array(X), np.array(y), ids


def customer_level_split(df: pd.DataFrame, test_size=0.15, val_size=0.15, random_state=None):
    """Splits customer IDs (not rows) into train/val/test, stratified by customer churn label."""
    random_state = random_state or config.RANDOM_STATE
    cust_labels = df.groupby(config.ID_COL)[config.TARGET_COL].max()

    # Cast to plain numpy/python objects (not pandas Index/Series) before handing
    # to scikit-learn: pandas 3.x can back string indexes with pyarrow, which
    # breaks sklearn's internal _safe_indexing with a TypeError.
    all_ids = np.array(cust_labels.index.tolist(), dtype=object)
    all_labels = cust_labels.to_numpy()

    train_ids, temp_ids, train_labels, temp_labels = train_test_split(
        all_ids,
        all_labels,
        test_size=(test_size + val_size),
        stratify=all_labels,
        random_state=random_state,
    )
    relative_test = test_size / (test_size + val_size)
    val_ids, test_ids = train_test_split(
        temp_ids,
        test_size=relative_test,
        stratify=temp_labels,
        random_state=random_state,
    )

    return set(train_ids), set(val_ids), set(test_ids)


if __name__ == "__main__":
    import os
    df = pd.read_csv(os.path.join(config.DATA_DIR, "featured_data.csv"))
    feature_cols = config.get_full_feature_list()
    X, y, ids = build_sequences(df, feature_cols)
    print("X shape:", X.shape, "y shape:", y.shape)
    train_ids, val_ids, test_ids = customer_level_split(df)
    print(f"Train customers: {len(train_ids)} | Val: {len(val_ids)} | Test: {len(test_ids)}")
