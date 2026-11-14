"""Dataset loader adapted from David Ciobanu's CS117 handoff."""

import io
import zipfile
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

LABELS = ("cardiomegaly", "pleural_effusion", "pneumothorax")


class ChestXrayDataset(Dataset):
    def __init__(self, csv_path, image_source, source_type="folder", transform=None):
        self.df = pd.read_csv(csv_path)
        self.image_source = Path(image_source)
        self.source_type = source_type
        self.transform = transform
        self.archive = None
        if source_type not in ("folder", "zip"):
            raise ValueError("source_type must be 'folder' or 'zip'")
        missing = {"image_path", *LABELS} - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing CSV columns: {sorted(missing)}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = str(row["image_path"])
        if self.source_type == "zip":
            if self.archive is None:
                self.archive = zipfile.ZipFile(self.image_source)
            with self.archive.open(path) as stream:
                image = Image.open(io.BytesIO(stream.read())).convert("RGB")
        else:
            with Image.open(self.image_source / path) as source:
                image = source.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        values = pd.to_numeric(row[list(LABELS)], errors="coerce").to_numpy(dtype=float)
        valid = (values == 0) | (values == 1)
        labels = torch.tensor([float(v) if ok else 0.0 for v, ok in zip(values, valid)], dtype=torch.float32)
        masks = torch.tensor(valid.astype(float), dtype=torch.float32)
        return image, labels, masks

    def __del__(self):
        archive = getattr(self, "archive", None)
        if archive is not None:
            archive.close()
