"""
Evaluate the trained LSTM on the held-out test split.

Run:
    python -m src.evaluate
"""

import os
import json
import numpy as np
from tensorflow import keras
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve,
)

from churn_src import config


def load_test_split():
    data = np.load(os.path.join(config.ARTIFACTS_DIR, "test_split.npz"))
    return data["X_test"], data["y_test"]


def evaluate(model=None, X_test=None, y_test=None, threshold: float = 0.5):
    if model is None:
        model = keras.models.load_model(config.MODEL_PATH)
    if X_test is None or y_test is None:
        X_test, y_test = load_test_split()

    y_prob = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob) if len(set(y_test)) > 1 else None,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, zero_division=0, output_dict=True),
        "threshold": threshold,
    }

    os.makedirs(config.ARTIFACTS_DIR, exist_ok=True)
    with open(config.METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print("=== Evaluation Metrics ===")
    for k in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
        print(f"{k}: {metrics[k]}")
    print("Confusion matrix:", metrics["confusion_matrix"])

    _save_plots(y_test, y_prob)
    return metrics


def _save_plots(y_test, y_prob):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

        if len(set(y_test)) > 1:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            axes[0].plot(fpr, tpr, label="LSTM")
            axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray")
            axes[0].set_title("ROC Curve")
            axes[0].set_xlabel("False Positive Rate")
            axes[0].set_ylabel("True Positive Rate")
            axes[0].legend()

            prec, rec, _ = precision_recall_curve(y_test, y_prob)
            axes[1].plot(rec, prec)
            axes[1].set_title("Precision-Recall Curve")
            axes[1].set_xlabel("Recall")
            axes[1].set_ylabel("Precision")

        plt.tight_layout()
        out_path = os.path.join(config.ARTIFACTS_DIR, "evaluation_curves.png")
        plt.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f"Saved evaluation curves -> {out_path}")
    except ImportError:
        print("matplotlib not installed, skipping plots.")


if __name__ == "__main__":
    evaluate()
