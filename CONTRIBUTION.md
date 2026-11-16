# CS117 team contributions

This file describes the files our team added to its fork of `pytorch/vision`. The original torchvision library and pretrained DenseNet-121 weights are from the upstream project, not our work.

| File added | Why it was added |
| --- | --- |
| `cs117_cross_hospital/dataset_loader.py` | Loads the team's CheXpert and NIH splits, returning images, three labels, and masks for missing or uncertain labels. Adapted from David Ciobanu's handoff. |
| `cs117_cross_hospital/prepare_data.py` | Extracts the prepared split CSVs and, when requested, NIH images from the Drive export ZIPs into a local, ignored data directory. |
| `cs117_cross_hospital/train.py` | Trains the CheXpert-only DenseNet-121 baseline and chooses a checkpoint using internal validation data. |
| `cs117_cross_hospital/evaluate.py` | Computes per-label and macro-AUROC on held-out CheXpert and NIH splits without fitting the model. |
| `cs117_cross_hospital/requirements.txt` | Lists the Python packages needed to run the baseline. |
| `cs117_cross_hospital/README.md` | Documents data placement, exact commands, evaluation boundaries, and current project scope. |
| `cs117_cross_hospital/.gitignore` | Prevents local images, split copies, checkpoints, logs, and virtual environments from being committed. |
| `cs117_cross_hospital/results/baseline.json` | Records the measured CheXpert and NIH baseline AUROCs and the source-to-target drop. |
| `CONTRIBUTION.md` | Separates team-written files from upstream torchvision code, as the course requires. |


## Noah Takashima: augmentation strengths and AdaBN

Adds the augmentation-strength comparison and AdaBN test-time adaptation on top of the baseline above.

| File | Added or changed | What it does |
| --- | --- | --- |
| `cs117_cross_hospital/augmentations.py` | Added | Defines three training augmentation strengths (`baseline`, `mild`, `strong`). All share the baseline rotation/translation, input size, and normalization; only brightness/contrast, Gaussian blur, and Gaussian noise strength differ. Noise is shared across RGB channels so X-rays stay grayscale. |
| `cs117_cross_hospital/adabn.py` | Added | Implements AdaBN (Li et al., 2016): recomputes every BatchNorm running mean/variance on 1,000 unlabeled NIH adaptation images while keeping all learned weights fixed, then freezes the statistics before evaluation. Labels are stripped before adaptation, and the code verifies no learned weight changed. |
| `cs117_cross_hospital/audit_augmentations.py` | Added | Audits the transforms on CheXpert source training images before training: saves example grids per strength and intensity statistics (clipping, grayscale check). |
| `cs117_cross_hospital/run_experiments.py` | Added | Runs the 3 strengths × 3 seeds (42, 43, 44) = 9 training runs with an identical training budget, evaluates each checkpoint with AdaBN OFF and ON (reusing the same weights), resumes after interruptions, and writes a mean ± std summary table. |
| `cs117_cross_hospital/tests/`, `cs117_cross_hospital/pytest.ini` | Added | 56 automated tests on synthetic data covering the augmentations, AdaBN (including an exact check of recomputed BN statistics), and end-to-end runs of every script. |
| `cs117_cross_hospital/train.py` | Changed | Added `--aug-level`, `--device`, and `--no-pretrained` (tests only); checkpoints now record the augmentation settings and training budget. Default behavior is unchanged (`baseline`). |
| `cs117_cross_hospital/evaluate.py` | Changed | Added `--adabn off/on/both` and AdaBN options. The default (`off`) output keeps the original `baseline.json` format. |
| `cs117_cross_hospital/README.md` | Changed | Documents the augmentation × AdaBN experiment, the real-image audit, and how to run the tests. |

**Augmentation audit on real images.** The strengths were checked on 200 real CheXpert training images before any training. `mild` kept anatomy clearly visible. The first `strong` setting (brightness/contrast ±0.30, blur kernel 7, noise σ=0.04) turned the heart and lower chest pure white in 15.5% of draws, which would hide cardiomegaly and effusion findings. `strong` was therefore reduced to brightness/contrast ±0.20, blur kernel 5 (σ ≤ 1.0), and noise σ=0.03, which cut those draws to 8%. Some already-bright films still saturate occasionally.

**Status.** The code and tests are complete; the nine training runs and AdaBN evaluations have not yet been run on the full data, so no augmentation or AdaBN results are reported yet.
