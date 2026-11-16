"""End-to-end runs of every script on the synthetic layout, invoked as in the README."""

import hashlib
import json
import re

import pytest
import torch

from conftest import run_script

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def trained(synthetic, tmp_path_factory):
    out = tmp_path_factory.mktemp("ckpt") / "strong_seed3"
    proc = run_script("train.py", "--chexpert-zip", synthetic["zip"], "--data-dir", synthetic["data"], "--output-dir", out,
                      "--aug-level", "strong", "--seed", 3, "--epochs", 2, "--batch-size", 8, "--no-pretrained")
    return out, proc


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------- train.py ----------

def test_train_writes_checkpoints_with_metadata(trained):
    out, proc = trained
    assert "aug_level=strong; seed=3" in proc.stdout
    assert {p.name for p in out.iterdir()} == {"epoch_1.pt", "epoch_2.pt", "best.pt"}
    ckpt = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
    assert ckpt["seed"] == 3 and ckpt["labels"] == ["cardiomegaly", "pleural_effusion", "pneumothorax"]
    assert ckpt["augmentation"]["level"] == "strong" and ckpt["augmentation"]["noise_std"] == 0.03
    assert ckpt["training_budget"] == {"epochs": 2, "batch_size": 8, "learning_rate": 1e-4}
    best_epoch_val = ckpt["validation"]["macro_auroc"]
    epochs = [json.loads(l) for l in proc.stdout.splitlines() if l.startswith('{"epoch"')]
    assert len(epochs) == 2
    assert best_epoch_val == max(e["validation"]["macro_auroc"] for e in epochs)


def test_train_default_level_is_baseline(synthetic, tmp_path):
    proc = run_script("train.py", "--chexpert-zip", synthetic["zip"], "--data-dir", synthetic["data"], "--output-dir", tmp_path,
                      "--epochs", 1, "--batch-size", 8, "--no-pretrained")
    assert "aug_level=baseline" in proc.stdout
    ckpt = torch.load(tmp_path / "best.pt", map_location="cpu", weights_only=False)
    assert ckpt["augmentation"]["level"] == "baseline" and "noise_std" not in ckpt["augmentation"]


def test_train_rejects_unknown_level(synthetic, tmp_path):
    proc = run_script("train.py", "--chexpert-zip", synthetic["zip"], "--aug-level", "huge", check=False)
    assert proc.returncode != 0 and "invalid choice" in proc.stderr


def test_same_seed_same_level_trains_identically(synthetic, tmp_path):
    """Seeds must make runs reproducible, so 3 seeds give 3 controlled replicates (checked on CPU)."""
    states = []
    for name in ("a", "b"):
        run_script("train.py", "--chexpert-zip", synthetic["zip"], "--data-dir", synthetic["data"], "--output-dir", tmp_path / name,
                   "--aug-level", "mild", "--seed", 5, "--epochs", 1, "--batch-size", 8, "--no-pretrained", "--device", "cpu")
        states.append(torch.load(tmp_path / name / "best.pt", map_location="cpu", weights_only=False)["model_state"])
    assert all(torch.equal(states[0][k], states[1][k]) for k in states[0])


def test_different_seeds_train_differently(synthetic, tmp_path):
    states = []
    for seed in (5, 6):
        run_script("train.py", "--chexpert-zip", synthetic["zip"], "--data-dir", synthetic["data"], "--output-dir", tmp_path / str(seed),
                   "--seed", seed, "--epochs", 1, "--batch-size", 8, "--no-pretrained", "--device", "cpu")
        states.append(torch.load(tmp_path / str(seed) / "best.pt", map_location="cpu", weights_only=False)["model_state"])
    assert not torch.equal(states[0]["classifier.weight"], states[1]["classifier.weight"])


# ---------- evaluate.py ----------

def evaluate(synthetic, ckpt, output, *extra, check=True):
    return run_script("evaluate.py", "--chexpert-zip", synthetic["zip"], "--data-dir", synthetic["data"],
                      "--checkpoint", ckpt, "--output", output, "--batch-size", 8, *extra, check=check)


def test_evaluate_default_keeps_baseline_format(synthetic, trained, tmp_path):
    out = tmp_path / "off.json"
    evaluate(synthetic, trained[0] / "best.pt", out, "--include-nih")
    r = json.loads(out.read_text())
    assert "adabn" not in r
    assert set(r["scores"]) == {"chexpert_internal_test", "chexpert_official_validation", "nih_test"}
    assert r["scores"]["nih_test"]["images"] == 24 and r["scores"]["chexpert_internal_test"]["images"] == 24
    assert set(r["scores"]["nih_test"]["per_label"]) == {"cardiomegaly", "pleural_effusion", "pneumothorax"}
    assert r["source_to_nih_macro_auroc_drop"] == pytest.approx(
        r["scores"]["chexpert_internal_test"]["macro_auroc"] - r["scores"]["nih_test"]["macro_auroc"])


def test_evaluate_adabn_both(synthetic, trained, tmp_path):
    ckpt = trained[0] / "best.pt"
    before = sha(ckpt)
    out = tmp_path / "both.json"
    evaluate(synthetic, ckpt, out, "--include-nih", "--adabn", "both", "--adabn-images", 40, "--adabn-batch-size", 10)
    assert sha(ckpt) == before, "evaluation modified the checkpoint"
    r = json.loads(out.read_text())
    a = r["adabn"]
    assert a["images"] == 40 and a["batches"] == 4 and a["bn_layers"] == 121
    assert set(a["scores"]) == {"nih_test"} and a["scores"]["nih_test"]["images"] == 24
    assert a["source_to_nih_macro_auroc_drop"] == pytest.approx(
        r["scores"]["chexpert_internal_test"]["macro_auroc"] - a["scores"]["nih_test"]["macro_auroc"])
    assert r["augmentation"]["level"] == "strong"


def test_evaluate_off_scores_unaffected_by_adabn(synthetic, trained, tmp_path):
    ckpt = trained[0] / "best.pt"
    evaluate(synthetic, ckpt, tmp_path / "off.json", "--include-nih")
    evaluate(synthetic, ckpt, tmp_path / "both.json", "--include-nih", "--adabn", "both", "--adabn-images", 40)
    off = json.loads((tmp_path / "off.json").read_text())["scores"]
    both = json.loads((tmp_path / "both.json").read_text())["scores"]
    for split in off:
        assert both[split]["macro_auroc"] == pytest.approx(off[split]["macro_auroc"], abs=1e-6)


def test_evaluate_adabn_on_only(synthetic, trained, tmp_path):
    out = tmp_path / "on.json"
    evaluate(synthetic, trained[0] / "best.pt", out, "--include-nih", "--adabn", "on", "--adabn-images", 40)
    r = json.loads(out.read_text())
    assert r["scores"] == {} and r["adabn"]["source_to_nih_macro_auroc_drop"] is not None


def test_evaluate_adabn_requires_nih(synthetic, trained, tmp_path):
    proc = evaluate(synthetic, trained[0] / "best.pt", tmp_path / "x.json", "--adabn", "on", check=False)
    assert proc.returncode != 0 and "needs --include-nih" in proc.stderr


def test_evaluate_adabn_reads_adaptation_split_not_test(synthetic, trained, tmp_path):
    """Pointing adaptation at the test folder must fail: a*.png names exist only in adaptation/."""
    proc = evaluate(synthetic, trained[0] / "best.pt", tmp_path / "x.json", "--include-nih", "--adabn", "on",
                    "--adabn-images", 40, "--nih-adaptation-dir", synthetic["data"] / "nih_images" / "test", check=False)
    assert proc.returncode != 0 and "a0.png" in proc.stderr


def test_evaluate_adabn_needs_enough_images(synthetic, trained, tmp_path):
    proc = evaluate(synthetic, trained[0] / "best.pt", tmp_path / "x.json", "--include-nih", "--adabn", "on", check=False)
    assert proc.returncode != 0 and "expected 1000 target images but the loader provided 40" in proc.stderr


# ---------- audit_augmentations.py ----------

def test_audit_writes_grids_and_stats_from_source_only(synthetic, tmp_path):
    import shutil
    source_only = tmp_path / "data"
    source_only.mkdir()
    shutil.copy(synthetic["data"] / "chexpert_train.csv", source_only)  # no NIH data present at all
    out = tmp_path / "audit"
    proc = run_script("audit_augmentations.py", "--chexpert-zip", synthetic["zip"], "--data-dir", source_only,
                      "--output-dir", out, "--grid-images", 3, "--draws", 4, "--stat-images", 20)
    assert {p.name for p in out.iterdir()} == {"baseline.png", "mild.png", "strong.png", "audit.json"}
    report = json.loads((out / "audit.json").read_text())
    assert report["source_images"] == 20
    for level in ("baseline", "mild", "strong"):
        s = report["levels"][level]
        assert s["max_channel_difference"] == 0 and 0 <= s["clipped_fraction"] < 0.2
    from PIL import Image
    w, h = Image.open(out / "strong.png").size
    assert w == 5 * 224 + 6 * 4 and h == 3 * 224 + 4 * 4  # 1 plain + 4 draws per row, 3 rows
    assert "strong:" in proc.stdout


# ---------- run_experiments.py ----------

def test_runner_dry_run_plans_nine_identical_budget_runs(synthetic, tmp_path):
    proc = run_script("run_experiments.py", "--chexpert-zip", synthetic["zip"], "--data-dir", synthetic["data"],
                      "--checkpoint-root", tmp_path / "c", "--results-dir", tmp_path / "r", "--dry-run")
    lines = [l for l in proc.stdout.splitlines() if l.startswith("+ ")]
    trains = [l for l in lines if "train.py" in l]
    evals = [l for l in lines if "evaluate.py" in l]
    assert len(trains) == 9 and len(evals) == 9
    pairs = {(re.search(r"--aug-level (\w+)", l).group(1), re.search(r"--seed (\d+)", l).group(1)) for l in trains}
    assert pairs == {(lv, s) for lv in ("baseline", "mild", "strong") for s in ("42", "43", "44")}
    budgets = {re.sub(r".*?(--epochs.*)", r"\1", l) for l in trains}
    assert len(budgets) == 1 and "--epochs 5 --batch-size 16 --learning-rate 0.0001" in budgets.pop()
    for l in evals:
        assert "--adabn both" in l and "--adabn-images 1000" in l and "--include-nih" in l
    assert not (tmp_path / "c").exists()


def test_runner_full_pipeline_and_resume(synthetic, tmp_path):
    args = ["--chexpert-zip", synthetic["zip"], "--data-dir", synthetic["data"], "--seeds", 1, "--epochs", 1,
            "--batch-size", 8, "--adabn-images", 40, "--no-pretrained",
            "--checkpoint-root", tmp_path / "ckpt", "--results-dir", tmp_path / "results" / "augmentation"]
    run_script("run_experiments.py", *args)
    for level in ("baseline", "mild", "strong"):
        r = json.loads((tmp_path / "results" / "augmentation" / f"{level}_seed1.json").read_text())
        assert r["augmentation"]["level"] == level and r["adabn"]["images"] == 40
        assert (tmp_path / "ckpt" / level / "seed_1" / "best.pt").exists()
    md = (tmp_path / "results" / "augmentation_summary.md").read_text()
    assert md.count("| baseline |") == 2 and md.count("| strong |") == 2 and "| ON |" in md
    summary = json.loads((tmp_path / "results" / "augmentation_summary.json").read_text())
    assert set(summary) == {f"{lv}/adabn_{c}" for lv in ("baseline", "mild", "strong") for c in ("off", "on")}
    # second invocation resumes: nothing retrained or re-evaluated
    again = run_script("run_experiments.py", *args)
    assert "+ " not in again.stdout and again.stdout.count("skip training") == 3 and again.stdout.count("skip evaluation") == 3
