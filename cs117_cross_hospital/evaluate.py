"""Evaluate a frozen CheXpert-trained checkpoint on held-out data, with optional AdaBN.

AdaBN OFF scores every split with the checkpoint as trained. AdaBN ON copies the
checkpoint, recomputes BatchNorm statistics on unlabeled NIH adaptation images
(default 1,000), freezes them, and scores the NIH test split. Learned weights
are identical in both conditions, so both reuse the same training run.
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from adabn import ImagesOnly, adapt_batchnorm
from augmentations import make_transform
from dataset_loader import ChestXrayDataset
from train import make_model, score


def drop(scores, source="chexpert_internal_test", target="nih_test"):
    s = scores.get(source, {}).get("macro_auroc")
    t = scores.get(target, {}).get("macro_auroc")
    return s - t if s is not None and t is not None else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chexpert-zip", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results" / "baseline.json")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--include-nih", action="store_true", help="Evaluate NIH after extracting its PNGs with prepare_data.py")
    parser.add_argument("--adabn", choices=("off", "on", "both"), default="off", help="'on'/'both' require --include-nih")
    parser.add_argument("--adabn-images", type=int, default=1000, help="Unlabeled NIH adaptation images used for BN statistics")
    parser.add_argument("--adabn-batch-size", type=int, default=50, help="Batch size during BN re-estimation")
    parser.add_argument("--nih-adaptation-dir", type=Path, default=None, help="Default: <data-dir>/nih_images/adaptation")
    args = parser.parse_args()
    if args.adabn != "off" and not args.include_nih:
        parser.error("--adabn on/both needs --include-nih")
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = make_model(pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()

    def loader(dataset, batch_size=args.batch_size):
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=args.workers)

    datasets = {
        "chexpert_internal_test": ChestXrayDataset(args.data_dir / "chexpert_test.csv", args.chexpert_zip, "zip", make_transform()),
        "chexpert_official_validation": ChestXrayDataset(args.data_dir / "chexpert_official_validation.csv", args.chexpert_zip, "zip", make_transform()),
    }
    if args.include_nih:
        datasets["nih_test"] = ChestXrayDataset(args.data_dir / "nih_test.csv", args.data_dir / "nih_images" / "test", "folder", make_transform())

    results = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint["epoch"],
        "seed": checkpoint["seed"],
        "augmentation": checkpoint.get("augmentation"),
        "scores": {},
    }
    if args.adabn in ("off", "both"):
        for name, dataset in datasets.items():
            results["scores"][name] = {"images": len(dataset), **score(model, loader(dataset), device)}
        results["source_to_nih_macro_auroc_drop"] = drop(results["scores"])

    if args.adabn in ("on", "both"):
        adapt_dir = args.nih_adaptation_dir or args.data_dir / "nih_images" / "adaptation"
        # Eval transform only (no augmentation); labels are stripped by ImagesOnly.
        adapt_data = ImagesOnly(ChestXrayDataset(args.data_dir / "nih_adaptation.csv", adapt_dir, "folder", make_transform()))
        adapted, info = adapt_batchnorm(model, loader(adapt_data, args.adabn_batch_size), device, args.adabn_images)
        nih = datasets["nih_test"]
        adabn = {**info, "scores": {"nih_test": {"images": len(nih), **score(adapted, loader(nih), device)}}}
        source = results["scores"].get("chexpert_internal_test")
        if source is None:  # adabn=on alone still needs the unadapted source score for the drop
            src = datasets["chexpert_internal_test"]
            source = {"images": len(src), **score(model, loader(src), device)}
        adabn["source_to_nih_macro_auroc_drop"] = drop({"chexpert_internal_test": source, **adabn["scores"]})
        results["adabn"] = adabn

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
