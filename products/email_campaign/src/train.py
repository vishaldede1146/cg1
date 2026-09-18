"""
Trains the tabular + text (DistilBERT) + image (EfficientNet) fusion
model for multi-output email-campaign performance prediction.

Run:
    python src/train.py
"""

import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.dataset import EmailCampaignDataset
from src.models.fusion_model import EmailPerformanceFusionModel


def get_target_weights(device):
    weights = torch.tensor(
        [config.TARGET_LOSS_WEIGHTS[t] for t in config.TARGET_COLUMNS],
        dtype=torch.float32,
    ).to(device)
    return weights


def weighted_mse_loss(pred, target, weights):
    per_target_mse = ((pred - target) ** 2).mean(dim=0)  # shape: [n_targets]
    return (per_target_mse * weights).sum() / weights.sum()


def run_epoch(model, loader, optimizer, weights, device, is_train=True):
    model.train() if is_train else model.eval()
    total_loss = 0.0
    n_batches = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch in tqdm(loader, desc="train" if is_train else "val", leave=False):
            tabular = batch["tabular"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            image = batch["image"].to(device)
            target = batch["target"].to(device)

            if is_train:
                optimizer.zero_grad()

            stacked_pred, _ = model(tabular, input_ids, attention_mask, image)
            loss = weighted_mse_loss(stacked_pred, target, weights)

            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1)


def main():
    device = config.DEVICE
    print(f"Using device: {device}")

    if not os.path.exists(config.TRAIN_CSV):
        raise FileNotFoundError(
            "Processed train/val/test CSVs not found. Run "
            "`python src/data_preprocessing.py` first (after generating the dataset)."
        )

    train_df = pd.read_csv(config.TRAIN_CSV)
    val_df = pd.read_csv(config.VAL_CSV)
    preprocessor = joblib.load(config.PREPROCESSOR_PATH)

    train_ds = EmailCampaignDataset(train_df, preprocessor, is_train=True)
    val_ds = EmailCampaignDataset(val_df, preprocessor, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
                               num_workers=config.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False,
                             num_workers=config.NUM_WORKERS, pin_memory=True)

    model = EmailPerformanceFusionModel(tabular_input_dim=preprocessor.output_dim).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=config.LEARNING_RATE,
                                   weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    weights = get_target_weights(device)

    best_val_loss = float("inf")
    patience_counter = 0
    history = []

    for epoch in range(1, config.NUM_EPOCHS + 1):
        start = time.time()
        train_loss = run_epoch(model, train_loader, optimizer, weights, device, is_train=True)
        val_loss = run_epoch(model, val_loader, optimizer, weights, device, is_train=False)
        scheduler.step(val_loss)
        elapsed = time.time() - start

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:02d}/{config.NUM_EPOCHS} | "
              f"train_loss={train_loss:.5f}  val_loss={val_loss:.5f}  "
              f"lr={current_lr:.2e}  ({elapsed:.1f}s)")

        history.append({"epoch": epoch, "train_loss": train_loss,
                         "val_loss": val_loss, "lr": current_lr})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "tabular_input_dim": preprocessor.output_dim,
                "target_names": config.TARGET_COLUMNS,
                "val_loss": val_loss,
                "epoch": epoch,
            }, config.BEST_MODEL_PATH)
            print(f"  -> New best model saved (val_loss={val_loss:.5f})")
        else:
            patience_counter += 1
            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    pd.DataFrame(history).to_csv(os.path.join(config.OUTPUTS_DIR, "training_history.csv"), index=False)
    print(f"\nTraining complete. Best val_loss={best_val_loss:.5f}")
    print(f"Best model saved to {config.BEST_MODEL_PATH}")


if __name__ == "__main__":
    main()
