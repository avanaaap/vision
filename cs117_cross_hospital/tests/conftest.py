"""Shared fixtures: a tiny synthetic CheXpert/NIH layout matching the real handoff."""

import io
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
FORK_ROOT = PROJECT.parent
sys.path.insert(0, str(PROJECT))

LABELS = ("cardiomegaly", "pleural_effusion", "pneumothorax")


def xray_like(rng, size=64, offset=0.0):
    """Grayscale image with smooth structure plus noise, so blur/contrast have something to act on."""
    y, x = np.mgrid[0:size, 0:size] / size
    base = 0.5 + 0.3 * np.sin(6 * x) * np.cos(4 * y) + offset
    a = np.clip(base + rng.normal(0, 0.05, (size, size)), 0, 1)
    return Image.fromarray((a * 255).astype("uint8"), "L")


def label_rows(rng, names, uncertain):
    rows = []
    for i, name in enumerate(names):
        lab = rng.integers(0, 2, 3).astype(float)
        if uncertain and i % 5 == 0:
            lab[0] = -1
        if uncertain and i % 7 == 0:
            lab[1] = np.nan
        rows.append({"image_path": name, **dict(zip(LABELS, lab))})
    return rows


@pytest.fixture(scope="session")
def synthetic(tmp_path_factory):
    """CheXpert splits inside a ZIP (like the real handoff) and NIH PNG folders.

    NIH adaptation and test files have different names/dirs (a*.png vs t*.png),
    so any accidental cross-use (e.g. adapting on test images) fails loudly.
    The target images are brighter, giving AdaBN a real shift to correct.
    """
    root = tmp_path_factory.mktemp("synthetic")
    data = root / "data"
    data.mkdir()
    rng = np.random.default_rng(0)
    zip_path = root / "chexpert.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        for split, n in [("chexpert_train", 48), ("chexpert_validation", 24), ("chexpert_test", 24), ("chexpert_official_validation", 16)]:
            rows = label_rows(rng, [f"CheXpert-v1.0-small/{split}/p{i}/view1_frontal.jpg" for i in range(n)], uncertain=True)
            pd.DataFrame(rows).to_csv(data / f"{split}.csv", index=False)
            for row in rows:
                buf = io.BytesIO()
                xray_like(rng).save(buf, "JPEG")
                archive.writestr(row["image_path"], buf.getvalue())
    for split, sub, prefix, n in [("nih_adaptation", "adaptation", "a", 40), ("nih_test", "test", "t", 24)]:
        rows = label_rows(rng, [f"{prefix}{i}.png" for i in range(n)], uncertain=False)
        pd.DataFrame(rows).to_csv(data / f"{split}.csv", index=False)
        (data / "nih_images" / sub).mkdir(parents=True)
        for row in rows:
            xray_like(rng, offset=0.2).save(data / "nih_images" / sub / row["image_path"])
    return {"root": root, "data": data, "zip": zip_path}


def run_script(name, *args, check=True):
    """Run a project script the way the README does: from the fork root."""
    cmd = [sys.executable, str(PROJECT / name), *map(str, args)]
    proc = subprocess.run(cmd, cwd=FORK_ROOT, capture_output=True, text=True)
    if check and proc.returncode != 0:
        raise AssertionError(f"{name} failed ({proc.returncode})\nSTDOUT:\n{proc.stdout[-3000:]}\nSTDERR:\n{proc.stderr[-3000:]}")
    return proc
