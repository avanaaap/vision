# CS117 cross-hospital chest X-ray baseline

This folder is the team's baseline experiment inside a fork of [`pytorch/vision`](https://github.com/pytorch/vision). It asks how much a DenseNet-121 classifier trained on CheXpert loses when evaluated on NIH ChestX-ray14. It reports AUROC separately for cardiomegaly, pleural effusion, and pneumothorax, plus their unweighted mean (macro-AUROC). The initial baseline uses standard small rotation and translation during CheXpert training. The folder also contains the augmentation-strength × AdaBN experiment (code and tests complete; full runs not yet done). See [Augmentation strength × AdaBN experiment](#augmentation-strength--adabn-experiment).

The starting model is upstream torchvision `densenet121` with `DenseNet121_Weights.IMAGENET1K_V1`. Our contribution is the chest X-ray data loader, training and evaluation workflow, data preparation, and results. The upstream model implementation and ImageNet weights are not ours.

## Data available for this baseline

The local handoff contains:

| Split | Images | Use |
| --- | ---: | --- |
| CheXpert training | 20,000 | Fit model weights |
| CheXpert internal validation | 2,000 | Choose training checkpoint |
| CheXpert internal test | 3,000 | Held-out source evaluation |
| CheXpert official validation | 202 frontal images | Separately reported source evaluation |
| NIH adaptation | 1,000 | AdaBN statistics only; labels are never used |
| NIH test | 5,000 | Held-out external evaluation |

The 3,000-image CheXpert internal test split comes from the CheXpert **training pool**. It is patient-disjoint from our training and internal validation splits, but it is **not the official CheXpert test set**. The NIH adaptation and test splits are also patient-disjoint. The current handoff has no VinDr-CXR or PadChest image data, so this starter evaluates only CheXpert and NIH. The team should report that scope change in the check-in and confirm it with the instructor.

CheXpert uncertain (`-1`) or missing labels are ignored for the affected finding. The loader returns a mask; training uses masked binary cross-entropy, and AUROC excludes masked labels. NIH labels are 0 or 1. The source and target datasets have different labeling processes, which is a limitation of cross-dataset comparisons.

## Data 

Clone this fork, then download the seven dataset ZIPs from the Google Drive to your computer. The commands below assume they are in your Downloads folder; change the paths if you saved them elsewhere.

## Install

Run from the root of your cloned fork. Python 3.10 or newer is recommended. Use a Python/PyTorch installation compatible with your operating system and GPU.

```bash
python3 -m venv cs117_cross_hospital/.venv
source cs117_cross_hospital/.venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r cs117_cross_hospital/requirements.txt
```

## Prepare local data

The script prepares the split CSVs and NIH images from the downloaded ZIPs. CheXpert stays zipped.

```bash
python cs117_cross_hospital/prepare_data.py \
  --downloads /Users/pavana/Downloads \
  --nih-images
```

If you are only training and checking CheXpert first, omit `--nih-images`. Add it before running NIH evaluation. On another computer, replace the Downloads path with that teammate's authorized ZIP location.

## Train the baseline

Run from the fork root. The first run downloads the upstream ImageNet weights if they are not already cached. The code uses the 20,000-row CheXpert training CSV and selects `best.pt` by **internal validation** macro-AUROC. It never uses the internal test, official validation, or NIH test to select a checkpoint.

```bash
python cs117_cross_hospital/train.py \
  --chexpert-zip /Users/pavana/Downloads/CheXpert-v1.0-small-001.zip \
  --epochs 5 --batch-size 16 --seed 42
```

The default is 5 epochs, learning rate `1e-4`, one seed, 224×224 inputs, and standard small rotation/translation. Change `--batch-size` if memory is limited. Save the console output locally if your course asks for training evidence. The `checkpoints/` folder is ignored by Git.

## Measure the baseline

```bash
python cs117_cross_hospital/evaluate.py \
  --chexpert-zip /Users/pavana/Downloads/CheXpert-v1.0-small-001.zip \
  --checkpoint cs117_cross_hospital/checkpoints/best.pt \
  --include-nih
```

This prints and saves AUROC for each finding and macro-AUROC on the CheXpert internal test, CheXpert official validation, and NIH test. It also computes `CheXpert internal test macro-AUROC - NIH test macro-AUROC`. The output is `cs117_cross_hospital/results/baseline.json`. The `results/` folder is ignored by default, so explicitly stage this small metrics file with `git add -f cs117_cross_hospital/results/baseline.json` when submitting results. If NIH images have not yet been extracted, omit `--include-nih` to evaluate CheXpert only.

The official model demo in the proposal established that torchvision's pretrained model could run. Its animal-classification output is **not** a measured chest X-ray baseline. Do not report a baseline AUROC until this evaluation has actually run.


## Augmentation strength × AdaBN experiment

Added by Noah Takashima (see `CONTRIBUTION.md`). Three augmentation strengths × three seeds = nine training runs on the same DenseNet-121 and training budget. Each run's `best.pt` is evaluated with **AdaBN OFF** and **AdaBN ON** (same weights, so no extra training).

| Level | Added to baseline geometry (±5° rotation, 2% translation) |
| --- | --- |
| `baseline` | nothing (AdaBN-OFF reference) |
| `mild` | brightness/contrast ±0.10 (p=0.8), blur k=3 σ∈[0.1,0.5] (p=0.3), noise σ=0.01 (p=0.5) |
| `strong` | brightness/contrast ±0.20 (p=0.8), blur k=5 σ∈[0.5,1.0] (p=0.5), noise σ=0.03 (p=0.5) — toned down from ±0.30 after auditing real CheXpert images (saturation) |

Settings live in `augmentations.py`. Noise is shared across RGB channels, so images stay grayscale.

**AdaBN** (`adabn.py`): copy the checkpoint, reset every BatchNorm running mean/variance, then forward 1,000 **unlabeled** NIH adaptation images through it with only the BN layers in train mode. This uses `torch.no_grad()`, an equal-weight cumulative average, and the eval transform. Then switch back to eval mode, which freezes the new statistics, and check that no learned weight changed. The adaptation dataset is wrapped so labels never reach this code.

### Real-image audit

The strengths were audited on 200 real CheXpert training images before training (`results/augmentation_audit/`, not committed):

| Level | Pixels clipped to pure black/white | Draws with >10% of the image newly white |
| --- | ---: | ---: |
| `baseline` | 3.1% (mostly black rotation corners) | — |
| `mild` | 3.4% | 1.7% |
| `strong` (original ±0.30, k=7, σ=0.04) | 6.5% | 15.5% |
| `strong` (current) | 5.1% | 8.0% |

`mild` keeps anatomy clearly visible. The original `strong` often turned the heart and lower chest pure white, hiding cardiomegaly and effusion, so it was toned down. Some already-bright films (often portable supine views) still saturate occasionally with the current `strong`; lowering brightness/contrast to ±0.15 would reduce that further but shrink the gap to `mild`.

### Running it

1. Audit the transforms on CheXpert source images first. Inspect the grids in `results/augmentation_audit/`:

```bash
python cs117_cross_hospital/audit_augmentations.py --chexpert-zip /path/to/CheXpert-v1.0-small-001.zip
```

2. Run all nine trainings plus the OFF/ON evaluations. The script resumes from finished runs; add `--dry-run` to only print the commands:

```bash
python cs117_cross_hospital/run_experiments.py --chexpert-zip /path/to/CheXpert-v1.0-small-001.zip --epochs 5 --batch-size 16
```

Outputs are `checkpoints/augmentation/<level>/seed_<seed>/`, per-run `results/augmentation/<level>_seed<seed>.json`, and `results/augmentation_summary.{json,md}` (mean ± std over seeds 42/43/44). NIH adaptation images are read from `data/nih_images/adaptation/` (override with `evaluate.py --nih-adaptation-dir`). A single checkpoint can also be evaluated with `evaluate.py --include-nih --adabn both`.

`train.py` also accepts `--device cpu` (or `cuda:0`, `mps`) to override automatic device selection.

## Run the tests

The tests use small synthetic data, so no real images are needed. They cover the augmentations, AdaBN (including an exact check of the recomputed BatchNorm statistics), and end-to-end runs of every script (about 2 minutes; add `-m "not slow"` for the 45-second unit tests only).

```bash
cd cs117_cross_hospital
python -m pip install pytest
python -m pytest
```
