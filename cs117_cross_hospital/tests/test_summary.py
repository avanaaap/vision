"""run_experiments.summarize: aggregation over seeds, checked against hand-computed numbers."""

import json

import numpy as np
import pytest

from run_experiments import mean_std, summarize


def fake_run(seed, src, nih_off, nih_on, level="mild"):
    per = lambda v: {"cardiomegaly": v, "pleural_effusion": v + 0.01, "pneumothorax": None}
    return {
        "seed": seed,
        "augmentation": {"level": level},
        "scores": {"chexpert_internal_test": {"macro_auroc": src}, "nih_test": {"macro_auroc": nih_off, "per_label": per(nih_off)}},
        "source_to_nih_macro_auroc_drop": src - nih_off,
        "adabn": {"images": 1000, "scores": {"nih_test": {"macro_auroc": nih_on, "per_label": per(nih_on)}},
                  "source_to_nih_macro_auroc_drop": src - nih_on},
    }


def test_mean_std():
    assert mean_std([]) is None and mean_std([None]) is None
    assert mean_std([0.5]) == {"mean": 0.5, "std": 0.0, "n": 1}
    r = mean_std([0.7, 0.8, None, 0.9])
    assert r["n"] == 3 and r["mean"] == pytest.approx(0.8) and r["std"] == pytest.approx(0.1)  # sample std


def test_summarize(tmp_path):
    results = tmp_path / "results" / "augmentation"
    results.mkdir(parents=True)
    runs = [(42, 0.89, 0.74, 0.78), (43, 0.88, 0.72, 0.77), (44, 0.90, 0.73, 0.79)]
    for seed, s, off, on in runs:
        (results / f"mild_seed{seed}.json").write_text(json.dumps(fake_run(seed, s, off, on)))
    summarize(["baseline", "mild", "strong"], [42, 43, 44], results)
    rows = json.loads((tmp_path / "results" / "augmentation_summary.json").read_text())
    assert set(rows) == {"mild/adabn_off", "mild/adabn_on"}  # levels without results are skipped
    off, on = rows["mild/adabn_off"], rows["mild/adabn_on"]
    assert off["seeds"] == [42, 43, 44]
    assert off["nih_test_macro"]["mean"] == pytest.approx(0.73)
    assert on["nih_test_macro"]["mean"] == pytest.approx(0.78)
    assert on["nih_test_macro"]["std"] == pytest.approx(np.std([0.78, 0.77, 0.79], ddof=1))
    assert off["chexpert_internal_test_macro"] == on["chexpert_internal_test_macro"]
    assert on["source_to_nih_drop"]["mean"] == pytest.approx(0.89 - 0.78)
    assert on["nih_test_per_label"]["pleural_effusion"]["mean"] == pytest.approx(0.79)
    assert on["nih_test_per_label"]["pneumothorax"] is None
    md = (tmp_path / "results" / "augmentation_summary.md").read_text()
    assert "| mild | ON | 3 | 0.890 ± 0.010 | 0.780 ± 0.010 |" in md and "| — |" in md


def test_summarize_partial_seeds(tmp_path):
    results = tmp_path / "results" / "augmentation"
    results.mkdir(parents=True)
    (results / "strong_seed42.json").write_text(json.dumps(fake_run(42, 0.85, 0.70, 0.75, "strong")))
    summarize(["strong"], [42, 43, 44], results)
    rows = json.loads((tmp_path / "results" / "augmentation_summary.json").read_text())
    assert rows["strong/adabn_on"]["seeds"] == [42] and rows["strong/adabn_on"]["nih_test_macro"]["n"] == 1
