"""
LSTM architecture for sequence-based churn classification.

Input:  (MAX_SEQ_LEN, n_features) per customer, zero-padded for short histories.
Output: single sigmoid probability of churn.

A Masking layer tells the LSTM to ignore the zero-padded timesteps so
customers with shorter histories aren't penalized.
"""

from tensorflow import keras
from tensorflow.keras import layers

from churn_src import config


def build_lstm_model(n_features: int, max_seq_len: int = None) -> keras.Model:
    max_seq_len = max_seq_len or config.MAX_SEQ_LEN

    inputs = keras.Input(shape=(max_seq_len, n_features), name="sequence_input")
    x = layers.Masking(mask_value=0.0)(inputs)

    x = layers.LSTM(config.LSTM_UNITS_1, return_sequences=True)(x)
    x = layers.Dropout(config.DROPOUT_RATE)(x)

    x = layers.LSTM(config.LSTM_UNITS_2, return_sequences=False)(x)
    x = layers.Dropout(config.DROPOUT_RATE)(x)

    x = layers.Dense(config.DENSE_UNITS, activation="relu")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="churn_probability")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="churn_lstm")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model


if __name__ == "__main__":
    m = build_lstm_model(n_features=len(config.get_full_feature_list()))
    m.summary()
