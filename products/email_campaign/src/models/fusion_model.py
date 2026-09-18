"""
Feature-fusion network: concatenates the tabular, text, and image
embeddings, passes them through a shared trunk, then predicts all six
targets with independent output heads (multi-output regression).
"""

import os
import sys

import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config
from src.models.tabular_model import TabularEncoder
from src.models.text_model import TextEncoder
from src.models.image_model import ImageEncoder


class EmailPerformanceFusionModel(nn.Module):
    def __init__(self, tabular_input_dim, target_names=config.TARGET_COLUMNS):
        super().__init__()
        self.target_names = target_names

        self.tabular_encoder = TabularEncoder(
            input_dim=tabular_input_dim,
            hidden_dims=config.TABULAR_HIDDEN_DIMS,
            out_dim=config.TABULAR_OUT_DIM,
        )
        self.text_encoder = TextEncoder(out_dim=config.TEXT_OUT_DIM)
        self.image_encoder = ImageEncoder(out_dim=config.IMAGE_OUT_DIM)

        fused_dim = (config.TABULAR_OUT_DIM + config.TEXT_OUT_DIM + config.IMAGE_OUT_DIM)

        self.fusion_trunk = nn.Sequential(
            nn.Linear(fused_dim, config.FUSION_HIDDEN_DIM),
            nn.BatchNorm1d(config.FUSION_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(config.FUSION_DROPOUT),
            nn.Linear(config.FUSION_HIDDEN_DIM, config.FUSION_HIDDEN_DIM // 2),
            nn.ReLU(),
        )

        trunk_out = config.FUSION_HIDDEN_DIM // 2
        # Independent linear head per target -> true multi-output prediction
        self.heads = nn.ModuleDict({
            name: nn.Linear(trunk_out, 1) for name in target_names
        })

    def encode(self, tabular, input_ids, attention_mask, image):
        tab_emb = self.tabular_encoder(tabular)
        text_emb = self.text_encoder(input_ids, attention_mask)
        img_emb = self.image_encoder(image)
        fused = torch.cat([tab_emb, text_emb, img_emb], dim=1)
        return fused, (tab_emb, text_emb, img_emb)

    def forward(self, tabular, input_ids, attention_mask, image):
        fused, _ = self.encode(tabular, input_ids, attention_mask, image)
        trunk_out = self.fusion_trunk(fused)
        outputs = {name: head(trunk_out).squeeze(-1) for name, head in self.heads.items()}
        # Also return stacked tensor in TARGET_COLUMNS order for convenient loss computation
        stacked = torch.stack([outputs[name] for name in self.target_names], dim=1)
        return stacked, outputs
