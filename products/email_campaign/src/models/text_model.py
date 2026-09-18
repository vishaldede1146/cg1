"""DistilBERT-based text encoder for subject + body copy."""

import os
import sys

import torch.nn as nn
from transformers import AutoModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config


class TextEncoder(nn.Module):
    def __init__(self, model_name=config.TEXT_MODEL_NAME, out_dim=64,
                 freeze_backbone=config.FREEZE_TEXT_BACKBONE, dropout=0.2):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.projection = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, out_dim),
        )
        self.out_dim = out_dim

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # Mean-pool over tokens (masked) instead of using only [CLS]
        token_embeddings = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = (token_embeddings * mask).sum(1)
        counts = mask.sum(1).clamp(min=1e-9)
        pooled = summed / counts
        return self.projection(pooled)
