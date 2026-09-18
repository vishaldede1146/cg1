"""
One-time conversion of the trained Keras LSTM churn model to ONNX.

This is NOT needed at serving time and should NOT run on the deployed
Render service. It's a local/offline step: run it once whenever you
retrain the model, commit the resulting .onnx file, and the deployed
API (predict_pipeline.py) loads that .onnx file via onnxruntime instead
of loading TensorFlow/Keras. This is what lets the deployed service
avoid installing the full tensorflow-cpu package (~500MB+), which is
one of the two biggest contributors to Render's 512MB OOM.

Run locally (needs tensorflow-cpu + tf2onnx, NOT part of requirements.txt):
    pip install tensorflow-cpu==2.20.0 tf2onnx onnxruntime
    python churn_src/export_onnx.py

Verified: outputs match the original Keras model within ~1e-7 (float32
precision), including the model's masking behavior on zero-padded
(shorter-than-12-month) sequences.
"""

import os
import sys

import numpy as np
import tensorflow as tf
import tf2onnx
from tensorflow import keras

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from churn_src import config


def export_to_onnx():
    print(f"Loading Keras model from {config.MODEL_PATH} ...")
    model = keras.models.load_model(config.MODEL_PATH)

    # Sequence shape: (batch, MAX_SEQ_LEN timesteps, n_features)
    n_features = model.input_shape[-1]
    max_seq_len = model.input_shape[1]
    spec = (tf.TensorSpec((None, max_seq_len, n_features), tf.float32, name="sequence_input"),)

    print(f"Exporting to ONNX (opset 17) -> {config.ONNX_MODEL_PATH} ...")
    tf2onnx.convert.from_keras(
        model, input_signature=spec, opset=17, output_path=config.ONNX_MODEL_PATH
    )
    print("Done.")
    return model


def verify_onnx(keras_model, atol=1e-4):
    import onnxruntime as ort

    n_features = keras_model.input_shape[-1]
    max_seq_len = keras_model.input_shape[1]

    rng = np.random.default_rng(0)
    X = rng.normal(size=(8, max_seq_len, n_features)).astype(np.float32)
    X[:, max_seq_len // 2:, :] = 0.0  # exercise the masking path too

    keras_out = keras_model.predict(X, verbose=0)

    sess = ort.InferenceSession(config.ONNX_MODEL_PATH, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    onnx_out = sess.run(None, {input_name: X})[0]

    max_diff = np.max(np.abs(keras_out - onnx_out))
    print(f"Max abs diff Keras vs ONNX: {max_diff:.8f} (tolerance={atol})")
    if max_diff > atol:
        print("WARNING: difference exceeds tolerance -- inspect before deploying.")
    else:
        print("ONNX export verified: outputs match Keras within tolerance.")


if __name__ == "__main__":
    keras_model = export_to_onnx()
    verify_onnx(keras_model)
