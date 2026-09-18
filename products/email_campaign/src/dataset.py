"""
Multimodal PyTorch Dataset that yields tabular features, tokenized text,
and a preprocessed image tensor together with the multi-output targets.
"""

import os
import sys

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import AutoTokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class EmailCampaignDataset(Dataset):
    def __init__(self, df: pd.DataFrame, preprocessor, tokenizer=None,
                 image_dir=config.DATA_DIR, is_train=True):
        self.df = df.reset_index(drop=True)
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)
        self.image_dir = image_dir
        self.is_train = is_train

        self.tabular_features = preprocessor.transform(self.df)
        self.targets = self.df[config.TARGET_COLUMNS].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def _load_image(self, rel_path):
        full_path = os.path.join(self.image_dir, rel_path)
        try:
            img = Image.open(full_path).convert("RGB")
        except (FileNotFoundError, OSError):
            img = Image.new("RGB", (config.IMAGE_SIZE, config.IMAGE_SIZE), color=(128, 128, 128))
        return IMAGE_TRANSFORM(img)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        tabular = torch.tensor(self.tabular_features[idx], dtype=torch.float32)

        text = f"{row['subject']} [SEP] {row['body']}"
        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=config.TEXT_MAX_LENGTH,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        image = self._load_image(row[config.IMAGE_COLUMN])

        target = torch.tensor(self.targets[idx], dtype=torch.float32)

        return {
            "tabular": tabular,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image": image,
            "target": target,
            "campaign_id": row[config.ID_COLUMN],
        }
