"""
TabularPreprocessor lives in its own module (never executed directly as
a script) so that joblib/pickle always resolves it via the stable dotted
path 'src.preprocessor.TabularPreprocessor', regardless of which script
is currently running as __main__.

(If this class were defined inside a script that gets run directly, e.g.
`python src/data_preprocessing.py`, pickle would save it under the module
name '__main__', and loading it later from a *different* __main__ script
like train.py would fail with:
    AttributeError: Can't get attribute 'TabularPreprocessor' on <module '__main__' ...>
Keeping it here avoids that entirely.)
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class TabularPreprocessor:
    """Wraps a StandardScaler for numeric features and a OneHotEncoder
    for categorical features so the exact same transform can be reused
    at both training and inference time."""

    def __init__(self):
        self.numeric_cols = config.NUMERIC_FEATURES + ["subject_length", "links_per_100_words"]
        self.categorical_cols = config.CATEGORICAL_FEATURES
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    def fit(self, df: pd.DataFrame):
        self.scaler.fit(df[self.numeric_cols])
        self.encoder.fit(df[self.categorical_cols])
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        num = self.scaler.transform(df[self.numeric_cols])
        cat = self.encoder.transform(df[self.categorical_cols])
        return np.concatenate([num, cat], axis=1).astype(np.float32)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        self.fit(df)
        return self.transform(df)

    @property
    def output_dim(self):
        return len(self.numeric_cols) + sum(len(c) for c in self.encoder.categories_)

    def feature_names(self):
        cat_names = self.encoder.get_feature_names_out(self.categorical_cols).tolist()
        return self.numeric_cols + cat_names
