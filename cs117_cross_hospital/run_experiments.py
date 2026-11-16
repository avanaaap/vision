"""Run the augmentation-strength x AdaBN experiment and summarize it.

Design: 3 augmentation levels x 3 seeds = 9 training runs, all with the same
architecture and training budget (epochs, batch size, learning rate). Each
run's best.pt (chosen by CheXpert internal-validation macro-AUROC) is evaluated
twice from the same weights: AdaBN OFF and AdaBN ON (BN statistics recomputed
on 1,000 unlabeled NIH adaptation images, then frozen).

Finished training runs and evaluations are skipped, so the script can be
re-run after an interruption. Use --summarize-only to rebuild the tables.

Outputs:
  checkpoints/augmentation/<level>/seed_<seed>/best.pt
  results/augmentation/<level>_seed<seed>.json     (per-run metrics)
  results/augmentation_summary.json / .md          (mean +/- std over seeds)
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from augmentations import LEVELS
from dataset_loader import LABELS

HERE = Path(__file__).parent


def run(cmd, dry_run):
    print("+ " + " ".join(str(c) for c in cmd), flush=True)
    if not dry_run:
        subprocess.run([str(c) for c in cmd], check=True)


def mean_std(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0, "n": len(values)}


def summarize(levels, seeds, results_dir):
    rows = {}
    for level in levels:
        runs = []
        for seed in seeds:
            path = results_dir / f"{level}_seed{seed}.json"
            if path.exists():
                runs.append(json.loads(path.read_text()))
        if not runs:
            continue
        for condition in ("off", "on"):
            def nih(r):
                block = r["scores"] if condition == "off" else r.get("adabn", {}).get("scores", {})
                return block.get("nih_test", {})

            def gap(r):
                return r.get("source_to_nih_macro_auroc_drop") if condition == "off" else r.get("adabn", {}).get("source_to_nih_macro_auroc_drop")

            rows[f"{level}/adabn_{condition}"] = {
                "level": level,
                "adabn": condition,
                "seeds": [r["seed"] for r in runs],
                "chexpert_internal_test_macro": mean_std([r["scores"].get("chexpert_internal_test", {}).get("macro_auroc") for r in runs]),
                "nih_test_macro": mean_std([nih(r).get("macro_auroc") for r in runs]),
                "nih_test_per_label": {name: mean_std([nih(r).get("per_label", {}).get(name) for r in runs]) for name in LABELS},
                "source_to_nih_drop": mean_std([gap(r) for r in runs]),
            }

    def fmt(cell):
        return "—" if cell is None else f"{cell['mean']:.3f} ± {cell['std']:.3f}"

    header = ["Augmentation", "AdaBN", "n", "CheXpert test macro", "NIH test macro", *[f"NIH {n}" for n in LABELS], "Drop"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in rows.values():
        lines.append("| " + " | ".join([
            row["level"], row["adabn"].upper(), str(len(row["seeds"])),
            fmt(row["chexpert_internal_test_macro"]), fmt(row["nih_test_macro"]),
            *[fmt(row["nih_test_per_label"][n]) for n in LABELS], fmt(row["source_to_nih_drop"]),
        ]) + " |")
    table = "\n".join(lines)
    note = ("Mean ± sample std over seeds. CheXpert test is always scored with source BN statistics; "
            "Drop = CheXpert internal test macro-AUROC − NIH test macro-AUROC.\n")
    (results_dir.parent / "augmentation_summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    (results_dir.parent / "augmentation_summary.md").write_text(table + "\n\n" + note)
    print(table)
    print(note)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chexpert-zip", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=HERE / "data")
    parser.add_argument("--levels", nargs="+", choices=sorted(LEVELS), default=["baseline", "mild", "strong"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--adabn-images", type=int, default=1000)
    parser.add_argument("--checkpoint-root", type=Path, default=HERE / "checkpoints" / "augmentation")
    parser.add_argument("--results-dir", type=Path, default=HERE / "results" / "augmentation")
    parser.add_argument("--no-pretrained", action="store_true", help="Random init (smoke tests only)")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    if not args.summarize_only:
        py = sys.executable
        for level in args.levels:
            for seed in args.seeds:
                ckpt_dir = args.checkpoint_root / level / f"seed_{seed}"
                if (ckpt_dir / f"epoch_{args.epochs}.pt").exists():
                    print(f"skip training {level} seed {seed}: {ckpt_dir} finished")
                else:
                    cmd = [py, HERE / "train.py", "--chexpert-zip", args.chexpert_zip, "--data-dir", args.data_dir,
                           "--output-dir", ckpt_dir, "--aug-level", level, "--seed", seed, "--epochs", args.epochs,
                           "--batch-size", args.batch_size, "--learning-rate", args.learning_rate, "--workers", args.workers]
                    run(cmd + (["--no-pretrained"] if args.no_pretrained else []), args.dry_run)
                out = args.results_dir / f"{level}_seed{seed}.json"
                if out.exists():
                    print(f"skip evaluation {level} seed {seed}: {out} exists")
                    continue
                run([py, HERE / "evaluate.py", "--chexpert-zip", args.chexpert_zip, "--data-dir", args.data_dir,
                     "--checkpoint", ckpt_dir / "best.pt", "--output", out, "--include-nih", "--adabn", "both",
                     "--adabn-images", args.adabn_images, "--batch-size", args.batch_size, "--workers", args.workers],
                    args.dry_run)
    if not args.dry_run:
        summarize(args.levels, args.seeds, args.results_dir)


if __name__ == "__main__":
    main()
