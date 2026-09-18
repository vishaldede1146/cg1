"""
SHAP explainability for the multimodal fusion model.

Two complementary explanations are produced:

1. Tabular feature attribution (SHAP KernelExplainer): for a sample of
   test instances, the tabular feature vector is perturbed while the
   *text* and *image* embeddings for that same instance are held fixed,
   isolating how much each tabular feature (audience size, discount,
   send time, personalization, ...) pushes the prediction.

2. Modality-level ablation importance: each modality's embedding is
   zeroed out in turn to measure how much the prediction shifts,
   giving a simple "how much does tabular vs text vs image matter"
   breakdown.

Run:
    python src/explain_shap.py
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd
import shap
import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.dataset import EmailCampaignDataset
from src.evaluate import load_model

TARGET_TO_EXPLAIN = "performance_score"
N_BACKGROUND = 50
N_EXPLAIN = 20


@torch.no_grad()
def compute_embeddings(model, batch, device):
    tabular = batch["tabular"].to(device)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    image = batch["image"].to(device)

    tab_emb = model.tabular_encoder(tabular)
    text_emb = model.text_encoder(input_ids, attention_mask)
    img_emb = model.image_encoder(image)
    return tab_emb, text_emb, img_emb


@torch.no_grad()
def predict_from_embeddings(model, tab_emb, text_emb, img_emb, target_idx, device):
    fused = torch.cat([tab_emb, text_emb, img_emb], dim=1).to(device)
    trunk_out = model.fusion_trunk(fused)
    target_name = model.target_names[target_idx]
    pred = model.heads[target_name](trunk_out).squeeze(-1)
    return pred.cpu().numpy()


def build_tabular_shap_wrapper(model, tabular_encoder_input_fn, fixed_text_emb, fixed_img_emb,
                                target_idx, device):
    """
    Returns a function f(X) -> np.ndarray[n] suitable for shap.KernelExplainer,
    where X is a (n_samples, n_tabular_raw_features) array of *preprocessed*
    tabular features (same space produced by TabularPreprocessor).
    text/image embeddings are broadcast (held fixed) across all perturbations.
    """
    def f(X):
        model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(device)
            tab_emb = model.tabular_encoder(X_t)
            n = tab_emb.shape[0]
            text_rep = fixed_text_emb.repeat(n, 1)
            img_rep = fixed_img_emb.repeat(n, 1)
            preds = predict_from_embeddings(model, tab_emb, text_rep, img_rep, target_idx, device)
        return preds
    return f


def modality_ablation_importance(model, tab_emb, text_emb, img_emb, target_idx, device):
    """Zero out each modality embedding in turn and measure the change
    in prediction relative to the full (all-modality) prediction."""
    full_pred = predict_from_embeddings(model, tab_emb, text_emb, img_emb, target_idx, device)

    zero_tab = torch.zeros_like(tab_emb)
    zero_text = torch.zeros_like(text_emb)
    zero_img = torch.zeros_like(img_emb)

    no_tab = predict_from_embeddings(model, zero_tab, text_emb, img_emb, target_idx, device)
    no_text = predict_from_embeddings(model, tab_emb, zero_text, img_emb, target_idx, device)
    no_img = predict_from_embeddings(model, tab_emb, text_emb, zero_img, target_idx, device)

    return {
        "tabular_importance": float(np.mean(np.abs(full_pred - no_tab))),
        "text_importance": float(np.mean(np.abs(full_pred - no_text))),
        "image_importance": float(np.mean(np.abs(full_pred - no_img))),
    }


def main():
    device = config.DEVICE
    model, checkpoint = load_model()
    target_idx = config.TARGET_COLUMNS.index(TARGET_TO_EXPLAIN)
    print(f"Explaining target: {TARGET_TO_EXPLAIN} (index {target_idx})")

    test_df = pd.read_csv(config.TEST_CSV).reset_index(drop=True)
    preprocessor = joblib.load(config.PREPROCESSOR_PATH)
    feature_names = preprocessor.feature_names()

    test_ds = EmailCampaignDataset(test_df, preprocessor, is_train=False)

    from torch.utils.data import DataLoader
    loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    batch = next(iter(loader))
    tab_emb_all, text_emb_all, img_emb_all = compute_embeddings(model, batch, device)
    tabular_raw_all = batch["tabular"].numpy()

    background = tabular_raw_all[:N_BACKGROUND]

    all_shap_rows = []
    ablation_rows = []

    n_explain = min(N_EXPLAIN, tabular_raw_all.shape[0])
    for i in range(n_explain):
        instance = tabular_raw_all[i:i + 1]
        fixed_text_emb = text_emb_all[i:i + 1]
        fixed_img_emb = img_emb_all[i:i + 1]
        fixed_tab_emb = tab_emb_all[i:i + 1]

        f = build_tabular_shap_wrapper(model, None, fixed_text_emb, fixed_img_emb, target_idx, device)

        explainer = shap.KernelExplainer(f, background, silent=True)
        shap_values = explainer.shap_values(instance, nsamples=100, silent=True)
        shap_values = np.array(shap_values).reshape(-1)

        row = {"campaign_id": test_df.loc[i, config.ID_COLUMN]}
        row.update({fname: val for fname, val in zip(feature_names, shap_values)})
        all_shap_rows.append(row)

        ablation = modality_ablation_importance(
            model, fixed_tab_emb, fixed_text_emb, fixed_img_emb, target_idx, device
        )
        ablation["campaign_id"] = test_df.loc[i, config.ID_COLUMN]
        ablation_rows.append(ablation)

        if (i + 1) % 5 == 0:
            print(f"Explained {i + 1}/{n_explain} instances...")

    shap_df = pd.DataFrame(all_shap_rows)
    shap_out_path = os.path.join(config.OUTPUTS_DIR, "shap_tabular_feature_attribution.csv")
    shap_df.to_csv(shap_out_path, index=False)
    print(f"Saved SHAP tabular attributions to {shap_out_path}")

    ablation_df = pd.DataFrame(ablation_rows)
    ablation_out_path = os.path.join(config.OUTPUTS_DIR, "modality_importance.csv")
    ablation_df.to_csv(ablation_out_path, index=False)
    print(f"Saved modality-level importance to {ablation_out_path}")

    # Global feature importance = mean absolute SHAP value per feature
    mean_abs_shap = shap_df[feature_names].abs().mean().sort_values(ascending=False)
    plt.figure(figsize=(8, 6))
    mean_abs_shap.head(15).iloc[::-1].plot(kind="barh")
    plt.title(f"Top Tabular Features Driving {TARGET_TO_EXPLAIN}")
    plt.xlabel("Mean |SHAP value|")
    plt.tight_layout()
    plot_path = os.path.join(config.OUTPUTS_DIR, "shap_feature_importance.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Saved SHAP importance plot to {plot_path}")

    print("\nModality-level average importance:")
    print(ablation_df[["tabular_importance", "text_importance", "image_importance"]].mean())


if __name__ == "__main__":
    main()
