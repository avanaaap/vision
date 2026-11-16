"""Training-time augmentation strengths for the CheXpert -> NIH experiment.

Every level shares the baseline geometry (small rotation/translation), input
size, and normalization. Only the photometric/degradation block changes, so the
comparison isolates brightness/contrast, noise, and blur strength.

Order inside a training transform:
    Resize -> RandomAffine -> [brightness/contrast] -> [Gaussian blur]
    -> ToTensor -> [Gaussian noise + clamp to [0, 1]] -> Normalize

Noise is drawn once per pixel and shared across the three channels, so the
grayscale X-ray stays grayscale after augmentation (audit_augmentations.py
checks this).
"""

import torch
from torchvision import transforms

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
IMAGE_SIZE = 224

GEOMETRY = {"degrees": 5, "translate": (0.02, 0.02)}

# p_* values are the probability each augmentation is applied to an image.
LEVELS = {
    "baseline": {
        "description": "standard small rotation/translation only (AdaBN-off reference)",
    },
    "mild": {
        "description": "baseline + mild brightness/contrast, noise, blur",
        "brightness": 0.10,
        "contrast": 0.10,
        "p_color": 0.8,
        "blur_kernel": 3,
        "blur_sigma": (0.1, 0.5),
        "p_blur": 0.3,
        "noise_std": 0.01,
        "p_noise": 0.5,
    },
    "strong": {
        "description": "baseline + stronger brightness/contrast, noise, blur",
        # Audited on 200 real CheXpert images: +/-0.30 brightness/contrast saturated the
        # heart/lower chest to pure white in 15.5% of draws; +/-0.20 cut that to 8%.
        "brightness": 0.20,
        "contrast": 0.20,
        "p_color": 0.8,
        "blur_kernel": 5,
        "blur_sigma": (0.5, 1.0),
        "p_blur": 0.5,
        "noise_std": 0.03,
        "p_noise": 0.5,
    },
}


class RandomGaussianNoise:
    """Add zero-mean Gaussian noise to a [0, 1] tensor image, then clamp."""

    def __init__(self, std, p):
        self.std = std
        self.p = p

    def __call__(self, image):
        if self.std <= 0 or torch.rand(()) >= self.p:
            return image
        noise = torch.randn((1, *image.shape[1:]), dtype=image.dtype) * self.std
        return (image + noise).clamp_(0.0, 1.0)

    def __repr__(self):
        return f"{type(self).__name__}(std={self.std}, p={self.p})"


def make_transform(training=False, level="baseline", normalize=True):
    """Build the transform. Evaluation (training=False) ignores ``level``."""
    if level not in LEVELS:
        raise ValueError(f"Unknown augmentation level {level!r}; choose from {sorted(LEVELS)}")
    cfg = LEVELS[level]
    pil_steps = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))]
    tensor_steps = [transforms.ToTensor()]
    if training:
        pil_steps.append(transforms.RandomAffine(**GEOMETRY))
        if cfg.get("p_color", 0) > 0:
            jitter = transforms.ColorJitter(brightness=cfg["brightness"], contrast=cfg["contrast"])
            pil_steps.append(transforms.RandomApply([jitter], p=cfg["p_color"]))
        if cfg.get("p_blur", 0) > 0:
            blur = transforms.GaussianBlur(cfg["blur_kernel"], sigma=cfg["blur_sigma"])
            pil_steps.append(transforms.RandomApply([blur], p=cfg["p_blur"]))
        if cfg.get("p_noise", 0) > 0:
            tensor_steps.append(RandomGaussianNoise(cfg["noise_std"], cfg["p_noise"]))
    if normalize:
        tensor_steps.append(transforms.Normalize(MEAN, STD))
    return transforms.Compose(pil_steps + tensor_steps)


def describe(level):
    """JSON-serializable record of a level, stored in checkpoints and results."""
    return {"level": level, "geometry": GEOMETRY, "image_size": IMAGE_SIZE, **LEVELS[level]}
