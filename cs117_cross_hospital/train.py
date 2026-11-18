"""Train the CheXpert-only DenseNet-121 at a chosen augmentation strength."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import DenseNet121_Weights, densenet121

from augmentations import LEVELS, describe, make_transform
from dataset_loader import ChestXrayDataset, LABELS

def make_model(pretrained=True):
    weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
    model = densenet121(weights=weights)
    model.classifier = nn.Linear(model.classifier.in_features, len(LABELS))
    return model


def auroc_summary(labels, probabilities, masks):
    per_label = {}
    for j, name in enumerate(LABELS):
        keep = masks[:, j].astype(bool)
        truth = labels[keep, j]
        if len(truth) == 0 or len(np.unique(truth)) != 2:
            per_label[name] = None
        else:
            per_label[name] = float(roc_auc_score(truth, probabilities[keep, j]))
    available = [value for value in per_label.values() if value is not None]
    return {"per_label": per_label, "macro_auroc": float(np.mean(available)) if available else None}


@torch.no_grad()
def score(model, loader, device):
    model.eval()
    all_labels, all_scores, all_masks = [], [], []
    for images, labels, masks in loader:
        logits = model(images.to(device))
        all_labels.append(labels.numpy())
        all_scores.append(torch.sigmoid(logits).cpu().numpy())
        all_masks.append(masks.numpy())
    return auroc_summary(np.concatenate(all_labels), np.concatenate(all_scores), np.concatenate(all_masks))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chexpert-zip", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "checkpoints")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--aug-level", choices=sorted(LEVELS), default="baseline", help="Training augmentation strength")
    parser.add_argument("--no-pretrained", action="store_true", help="Random init (smoke tests only)")
    parser.add_argument("--device", default="auto", help="auto (cuda > mps > cpu), or e.g. cpu, cuda:0")
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device != "auto":
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    train_data = ChestXrayDataset(args.data_dir / "chexpert_train.csv", args.chexpert_zip, "zip", make_transform(True, args.aug_level))
    val_data = ChestXrayDataset(args.data_dir / "chexpert_validation.csv", args.chexpert_zip, "zip", make_transform())
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    model = make_model(pretrained=not args.no_pretrained).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best = float("-inf")
    print(f"device={device}; aug_level={args.aug_level}; seed={args.seed}; train={len(train_data)}; validation={len(val_data)}; labels={LABELS}", flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, batches = 0.0, 0
        for images, labels, masks in train_loader:
            images, labels, masks = images.to(device), labels.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = (criterion(logits, labels) * masks).sum() / masks.sum().clamp_min(1)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batches += 1
        validation = score(model, val_loader, device)
        checkpoint = {
            "model_state": model.state_dict(),
            "epoch": epoch,
            "seed": args.seed,
            "validation": validation,
            "labels": list(LABELS),
            "augmentation": describe(args.aug_level),
            "training_budget": {"epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": args.learning_rate},
        }
        torch.save(checkpoint, args.output_dir / f"epoch_{epoch}.pt")
        value = validation["macro_auroc"]
        if value is not None and value > best:
            best = value
            torch.save(checkpoint, args.output_dir / "best.pt")
        print(json.dumps({"epoch": epoch, "train_masked_bce": total_loss / batches, "validation": validation}), flush=True)
    print(f"best_validation_macro_auroc={best:.4f}; checkpoint={args.output_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
