"""
Evaluates the trained fusion model on the held-out test set and reports
per-target regression metrics (MAE, RMSE, R^2).

Run:
    python src/evaluate.py
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.dataset import EmailCampaignDataset
from src.models.fusion_model import EmailPerformanceFusionModel


def load_model(checkpoint_path=config.BEST_MODEL_PATH, device=config.DEVICE):
    # PyTorch >= 2.6 defaults torch.load(weights_only=True), which only allows
    # plain tensors/primitives through its unpickler. This checkpoint is a dict
    # bundling model_state_dict alongside training metadata (tabular_input_dim,
    # etc.), so the safe unpickler rejects it. We trust this checkpoint (it's
    # our own training output, not an untrusted download), so we explicitly
    # opt back into the old, permissive behavior.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = EmailPerformanceFusionModel(tabular_input_dim=checkpoint["tabular_input_dim"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def evaluate(model, loader, device):
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="evaluating"):
            tabular = batch["tabular"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            image = batch["image"].to(device)
            target = batch["target"].to(device)

            stacked_pred, _ = model(tabular, input_ids, attention_mask, image)
            all_preds.append(stacked_pred.cpu().numpy())
            all_targets.append(target.cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    return preds, targets


def report_metrics(preds, targets, target_names=config.TARGET_COLUMNS):
    rows = []
    for i, name in enumerate(target_names):
        y_true = targets[:, i]
        y_pred = preds[:, i]
        mae = mean_absolute_error(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred) ** 0.5
        r2 = r2_score(y_true, y_pred)
        rows.append({"target": name, "MAE": mae, "RMSE": rmse, "R2": r2})
        print(f"{name:20s} | MAE={mae:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}")
    return pd.DataFrame(rows)


def main():
    device = config.DEVICE
    test_df = pd.read_csv(config.TEST_CSV)
    preprocessor = joblib.load(config.PREPROCESSOR_PATH)

    test_ds = EmailCampaignDataset(test_df, preprocessor, is_train=False)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False,
                              num_workers=config.NUM_WORKERS)

    model, checkpoint = load_model()
    print(f"Loaded model from epoch {checkpoint['epoch']} (val_loss={checkpoint['val_loss']:.5f})")

    preds, targets = evaluate(model, test_loader, device)
    metrics_df = report_metrics(preds, targets)

    metrics_path = os.path.join(config.OUTPUTS_DIR, "test_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nSaved test metrics to {metrics_path}")

    preds_df = pd.DataFrame(preds, columns=[f"pred_{t}" for t in config.TARGET_COLUMNS])
    targets_df = pd.DataFrame(targets, columns=[f"true_{t}" for t in config.TARGET_COLUMNS])
    out_df = pd.concat([test_df[[config.ID_COLUMN]].reset_index(drop=True), preds_df, targets_df], axis=1)
    predictions_path = os.path.join(config.OUTPUTS_DIR, "test_predictions.csv")
    out_df.to_csv(predictions_path, index=False)
    print(f"Saved test predictions to {predictions_path}")


if __name__ == "__main__":
    main()
