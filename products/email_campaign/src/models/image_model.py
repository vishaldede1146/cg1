"""EfficientNet-B0-based encoder for the email creative image."""

import os
import sys

import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config


class ImageEncoder(nn.Module):
    def __init__(self, out_dim=64, freeze_backbone=config.FREEZE_IMAGE_BACKBONE, dropout=0.2):
        super().__init__()
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        backbone = efficientnet_b0(weights=weights)

        # Keep the convolutional feature extractor + pooling, drop the classifier
        self.features = backbone.features
        self.pool = backbone.avgpool
        in_features = backbone.classifier[1].in_features  # 1280 for b0

        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False

        self.projection = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, out_dim),
        )
        self.out_dim = out_dim

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.flatten(1)
        return self.projection(x)
