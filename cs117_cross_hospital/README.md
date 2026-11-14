# CS117 cross-hospital chest X-ray baseline

This folder is the team's baseline experiment inside a fork of [`pytorch/vision`](https://github.com/pytorch/vision). It asks how much a DenseNet-121 classifier trained on CheXpert loses when evaluated on NIH ChestX-ray14. It reports AUROC separately for cardiomegaly, pleural effusion, and pneumothorax, plus their unweighted mean (macro-AUROC). The initial baseline uses standard small rotation and translation during CheXpert training. It does not yet run AdaBN or compare augmentation strengths.

The starting model is upstream torchvision `densenet121` with `DenseNet121_Weights.IMAGENET1K_V1`. Our contribution is the chest X-ray data loader, training and evaluation workflow, data preparation, and results. The upstream model implementation and ImageNet weights are not ours.

## Data available for this baseline

The local handoff contains:

| Split | Images | Use |
| --- | ---: | --- |
| CheXpert training | 20,000 | Fit model weights |
| CheXpert internal validation | 2,000 | Choose training checkpoint |
| CheXpert internal test | 3,000 | Held-out source evaluation |
| CheXpert official validation | 202 frontal images | Separately reported source evaluation |
| NIH adaptation | 1,000 | Reserved for later AdaBN; labels must not be used for adaptation |
| NIH test | 5,000 | Held-out external evaluation |

The 3,000-image CheXpert internal test split comes from the CheXpert **training pool**. It is patient-disjoint from our training and internal validation splits, but it is **not the official CheXpert test set**. The NIH adaptation and test splits are also patient-disjoint. The current handoff has no VinDr-CXR or PadChest image data, so this starter evaluates only CheXpert and NIH. The team should report that scope change in the check-in and confirm it with the instructor.

CheXpert uncertain (`-1`) or missing labels are ignored for the affected finding. The loader returns a mask; training uses masked binary cross-entropy, and AUROC excludes masked labels. NIH labels are 0 or 1. The source and target datasets have different labeling processes, which is a limitation of cross-dataset comparisons.

## Data for teammates

Clone this fork, then download the seven dataset ZIPs from the team's Google Drive to your computer. The commands below assume they are in your Downloads folder; change the paths if you saved them elsewhere.

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

