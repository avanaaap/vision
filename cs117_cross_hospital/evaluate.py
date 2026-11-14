"""Evaluate a frozen CheXpert-trained checkpoint on held-out data."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset_loader import ChestXrayDataset
from train import make_model, make_transform, score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chexpert-zip", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results" / "baseline.json")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--include-nih", action="store_true", help="Evaluate NIH after extracting its PNGs with prepare_data.py")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = make_model(pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    datasets = {
        "chexpert_internal_test": ChestXrayDataset(args.data_dir / "chexpert_test.csv", args.chexpert_zip, "zip", make_transform()),
        "chexpert_official_validation": ChestXrayDataset(args.data_dir / "chexpert_official_validation.csv", args.chexpert_zip, "zip", make_transform()),
    }
    if args.include_nih:
        datasets["nih_test"] = ChestXrayDataset(args.data_dir / "nih_test.csv", args.data_dir / "nih_images" / "test", "folder", make_transform())
    results = {"checkpoint_epoch": checkpoint["epoch"], "seed": checkpoint["seed"], "scores": {}}
    for name, dataset in datasets.items():
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        results["scores"][name] = {"images": len(dataset), **score(model, loader, device)}
    source = results["scores"]["chexpert_internal_test"]["macro_auroc"]
    target = results["scores"].get("nih_test", {}).get("macro_auroc")
    if source is not None and target is not None:
        results["source_to_nih_macro_auroc_drop"] = source - target
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
