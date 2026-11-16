"""Audit augmentation levels on CheXpert *source* training images before training.

For each level this writes a grid of augmented images (row = one source image:
the unaugmented eval view, then several random draws) and intensity statistics
in [0, 1] pixel space (before normalization):

- mean / std intensity, compared with the unaugmented eval transform
- fraction of pixels clipped to exactly 0 or 1
- max |R-G|, |G-B| difference (must be 0: augmentations keep X-rays grayscale)

Look at the grids: strong augmentation should still look like a plausible chest
X-ray with visible anatomy. Uses only CheXpert training images; no NIH data.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torchvision.utils import save_image

from augmentations import LEVELS, describe, make_transform
from dataset_loader import ChestXrayDataset


def stats(images):
    x = torch.stack(images)
    return {
        "mean_intensity": float(x.mean()),
        "std_intensity": float(x.std()),
        "clipped_fraction": float(((x <= 0) | (x >= 1)).float().mean()),
        "max_channel_difference": float(max((x[:, 0] - x[:, 1]).abs().max(), (x[:, 1] - x[:, 2]).abs().max())),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chexpert-zip", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "results" / "augmentation_audit")
    parser.add_argument("--grid-images", type=int, default=6, help="Source images shown per grid")
    parser.add_argument("--draws", type=int, default=5, help="Random augmentations per grid image")
    parser.add_argument("--stat-images", type=int, default=200, help="Source images used for statistics")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw = ChestXrayDataset(args.data_dir / "chexpert_train.csv", args.chexpert_zip, "zip", transform=None)
    rng = random.Random(args.seed)
    indices = rng.sample(range(len(raw)), min(len(raw), max(args.grid_images, args.stat_images)))
    sources = [raw[i][0] for i in indices]  # PIL images; labels unused
    plain = make_transform(training=False, normalize=False)

    report = {"source_images": len(sources), "seed": args.seed, "eval_transform": stats([plain(im) for im in sources]), "levels": {}}
    for level in LEVELS:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        aug = make_transform(training=True, level=level, normalize=False)
        grid = []
        for image in sources[: args.grid_images]:
            grid.append(plain(image))
            grid.extend(aug(image) for _ in range(args.draws))
        grid_path = args.output_dir / f"{level}.png"
        save_image(torch.stack(grid), grid_path, nrow=args.draws + 1, padding=4, pad_value=1.0)
        report["levels"][level] = {
            "config": describe(level),
            "grid": str(grid_path),
            **stats([aug(im) for im in sources[: args.stat_images]]),
        }
        level_stats = report["levels"][level]
        if level_stats["max_channel_difference"] > 1e-6:
            raise SystemExit(f"{level}: augmentation broke grayscale (channel difference {level_stats['max_channel_difference']})")
        print(f"{level:>8}: mean={level_stats['mean_intensity']:.3f} std={level_stats['std_intensity']:.3f} "
              f"clipped={level_stats['clipped_fraction']:.4f} grid={grid_path}")

    out = args.output_dir / "audit.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"eval reference: {report['eval_transform']}")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
