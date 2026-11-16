import json
import random

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision import transforms

from augmentations import GEOMETRY, LEVELS, MEAN, STD, RandomGaussianNoise, describe, make_transform
from conftest import xray_like


@pytest.fixture
def image():
    return xray_like(np.random.default_rng(1), size=96).convert("RGB")


def kinds(transform):
    """Flatten Compose/RandomApply into the list of transform classes."""
    out = []
    for t in transform.transforms:
        if isinstance(t, transforms.RandomApply):
            out.extend(type(inner) for inner in t.transforms)
        else:
            out.append(type(t))
    return out


def seed_all(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def test_three_levels_exist():
    assert set(LEVELS) == {"baseline", "mild", "strong"}


def test_unknown_level_rejected():
    with pytest.raises(ValueError, match="Unknown augmentation level"):
        make_transform(True, "extreme")


def test_baseline_matches_original_pipeline(image):
    """Baseline must be exactly the pre-existing train.py transform."""
    original = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomAffine(degrees=5, translate=(0.02, 0.02)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    ours = make_transform(True, "baseline")
    assert kinds(ours) == kinds(original)
    seed_all(3)
    a = ours(image)
    seed_all(3)
    b = original(image)
    assert torch.equal(a, b)


@pytest.mark.parametrize("level", sorted(LEVELS))
def test_eval_transform_ignores_level(image, level):
    assert torch.equal(make_transform(False, level)(image), make_transform(False)(image))
    assert kinds(make_transform(False, level)) == [transforms.Resize, transforms.ToTensor, transforms.Normalize]


@pytest.mark.parametrize("level", sorted(LEVELS))
def test_every_level_keeps_shared_geometry_and_normalization(level):
    t = make_transform(True, level)
    affine = [s for s in t.transforms if isinstance(s, transforms.RandomAffine)]
    assert len(affine) == 1
    assert list(affine[0].degrees) == [-GEOMETRY["degrees"], GEOMETRY["degrees"]]
    assert tuple(affine[0].translate) == GEOMETRY["translate"]
    assert isinstance(t.transforms[0], transforms.Resize) and t.transforms[0].size == (224, 224)
    assert isinstance(t.transforms[-1], transforms.Normalize)


@pytest.mark.parametrize("level", ["mild", "strong"])
def test_photometric_levels_contain_all_three_augmentations(level):
    k = kinds(make_transform(True, level))
    assert transforms.ColorJitter in k and transforms.GaussianBlur in k and RandomGaussianNoise in k
    # noise is applied in [0, 1] space, before normalization
    assert k.index(RandomGaussianNoise) < k.index(transforms.Normalize)
    assert k.index(transforms.ToTensor) < k.index(RandomGaussianNoise)


def test_strong_is_at_least_as_strong_as_mild_on_every_knob():
    mild, strong = LEVELS["mild"], LEVELS["strong"]
    for key in ("brightness", "contrast", "blur_kernel", "noise_std", "p_color", "p_blur", "p_noise"):
        assert strong[key] >= mild[key], key
    assert strong["blur_sigma"][1] > mild["blur_sigma"][1]
    assert strong["noise_std"] > mild["noise_std"] and strong["brightness"] > mild["brightness"]


@pytest.mark.parametrize("level", sorted(LEVELS))
def test_output_shape_dtype_and_grayscale(image, level):
    t = make_transform(True, level, normalize=False)
    seed_all(0)
    for _ in range(30):
        x = t(image)
        assert x.shape == (3, 224, 224) and x.dtype == torch.float32
        assert x.min() >= 0 and x.max() <= 1
        assert torch.equal(x[0], x[1]) and torch.equal(x[1], x[2]), "augmentation broke grayscale"


@pytest.mark.parametrize("level", sorted(LEVELS))
def test_same_seed_is_reproducible(image, level):
    t = make_transform(True, level)
    seed_all(7)
    a = [t(image) for _ in range(5)]
    seed_all(7)
    b = [t(image) for _ in range(5)]
    assert all(torch.equal(x, y) for x, y in zip(a, b))


def photometric_change(level, image, draws=40):
    """Mean |augmented - plain| with geometry held fixed, isolating the photometric block."""
    cfg = LEVELS[level]
    seed_all(0)
    plain = make_transform(False, normalize=False)(image)
    steps = []
    if cfg.get("p_color"):
        steps.append(transforms.RandomApply([transforms.ColorJitter(cfg["brightness"], cfg["contrast"])], cfg["p_color"]))
    if cfg.get("p_blur"):
        steps.append(transforms.RandomApply([transforms.GaussianBlur(cfg["blur_kernel"], cfg["blur_sigma"])], cfg["p_blur"]))
    pil = transforms.Compose([transforms.Resize((224, 224)), *steps, transforms.ToTensor()])
    noise = RandomGaussianNoise(cfg.get("noise_std", 0), cfg.get("p_noise", 0))
    return float(np.mean([(noise(pil(image)) - plain).abs().mean() for _ in range(draws)]))


def test_strength_ordering_measured_on_images(image):
    base, mild, strong = (photometric_change(lv, image) for lv in ("baseline", "mild", "strong"))
    assert base == 0
    assert 0 < mild < strong


def test_noise_disabled_is_identity():
    x = torch.rand(3, 8, 8)
    assert RandomGaussianNoise(0.1, p=0.0)(x) is x
    assert RandomGaussianNoise(0.0, p=1.0)(x) is x


def test_noise_statistics_and_clamping():
    torch.manual_seed(0)
    x = torch.full((3, 256, 256), 0.5)
    y = RandomGaussianNoise(0.04, p=1.0)(x.clone())
    diff = y - x
    assert abs(float(diff.mean())) < 2e-3
    assert abs(float(diff.std()) - 0.04) < 2e-3
    assert torch.equal(y[0], y[1])
    edge = RandomGaussianNoise(0.5, p=1.0)(torch.ones(3, 64, 64))
    assert edge.min() >= 0 and edge.max() == 1


def test_noise_probability():
    torch.manual_seed(0)
    noise = RandomGaussianNoise(0.1, p=0.5)
    x = torch.full((1, 4, 4), 0.5)
    applied = sum(not torch.equal(noise(x.clone()), x) for _ in range(2000))
    assert 900 < applied < 1100


@pytest.mark.parametrize("level", sorted(LEVELS))
def test_describe_is_json_serializable(level):
    d = json.loads(json.dumps(describe(level)))
    assert d["level"] == level and d["image_size"] == 224


def test_dataset_masks_uncertain_and_missing(synthetic):
    from dataset_loader import ChestXrayDataset

    ds = ChestXrayDataset(synthetic["data"] / "chexpert_train.csv", synthetic["zip"], "zip", make_transform(True, "strong"))
    image, labels, masks = ds[0]  # row 0: cardiomegaly = -1 and pleural_effusion = NaN
    assert image.shape == (3, 224, 224)
    assert masks.tolist() == [0.0, 0.0, 1.0] and labels[0] == 0 and labels[1] == 0
