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

Augmentation-strength experiments and AdaBN are not implemented here; list any files added for them before final submission.
