import copy

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from adabn import ImagesOnly, adapt_batchnorm
from train import make_model


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    m = make_model(pretrained=False)
    # Give it non-trivial "source" statistics, like a trained checkpoint would have.
    m.train()
    with torch.no_grad():
        for _ in range(3):
            m(torch.randn(8, 3, 64, 64))
    return m.eval()


@pytest.fixture(scope="module")
def target():
    torch.manual_seed(1)
    return torch.randn(60, 3, 64, 64) * 1.5 + 0.7  # shifted/scaled "target hospital"


def loader(images, batch_size=10, shuffle=False):
    return DataLoader(ImagesOnly(TensorDataset(images, torch.zeros(len(images), 3))), batch_size=batch_size, shuffle=shuffle)


def bn_stats(m):
    return {k: v.clone() for k, v in m.state_dict().items() if "running_" in k}


def test_images_only_strips_labels():
    ds = ImagesOnly(TensorDataset(torch.ones(4, 3, 2, 2), torch.zeros(4, 3), torch.zeros(4, 3)))
    assert len(ds) == 4
    item = ds[0]
    assert torch.is_tensor(item) and item.shape == (3, 2, 2)


def test_learned_weights_identical_and_source_model_untouched(model, target):
    before_params = copy.deepcopy(model.state_dict())
    adapted, info = adapt_batchnorm(model, loader(target), "cpu", max_images=40)
    # original checkpoint unchanged (all params and buffers)
    for k, v in model.state_dict().items():
        assert torch.equal(v, before_params[k]), k
    # adapted: every learned parameter (incl. BN gamma/beta) identical
    for name, p in adapted.named_parameters():
        assert torch.equal(p, dict(model.named_parameters())[name]), name
    assert info == {"images": 40, "batches": 4, "bn_layers": 121}


def test_only_bn_running_stats_change(model, target):
    adapted, _ = adapt_batchnorm(model, loader(target), "cpu", max_images=40)
    old, new = model.state_dict(), adapted.state_dict()
    changed = {k for k in old if not torch.equal(old[k], new[k])}
    assert changed, "AdaBN changed nothing"
    assert all(("running_mean" in k or "running_var" in k or "num_batches_tracked" in k) for k in changed), changed
    assert sum("running_mean" in k for k in changed) == 121


def test_first_bn_mean_equals_exact_target_mean(model, target):
    """norm0 directly follows conv0, so its new running mean must equal the conv0 output channel means."""
    images = target[:40]
    adapted, _ = adapt_batchnorm(model, loader(images, batch_size=10), "cpu", max_images=40)
    with torch.no_grad():
        conv = model.features.conv0(images)
    expected_mean = conv.mean(dim=(0, 2, 3))
    torch.testing.assert_close(adapted.features.norm0.running_mean, expected_mean, rtol=1e-4, atol=1e-5)
    # variance: cumulative average of per-batch unbiased variances (equal batch sizes)
    per_batch = torch.stack([conv[i:i + 10].var(dim=(0, 2, 3), unbiased=True) for i in range(0, 40, 10)])
    torch.testing.assert_close(adapted.features.norm0.running_var, per_batch.mean(0), rtol=1e-4, atol=1e-5)
    assert not torch.allclose(adapted.features.norm0.running_mean, model.features.norm0.running_mean)


def test_uses_exactly_max_images(model, target):
    """With 60 available and max_images=40, stats come from the first 40 only."""
    from_60, info = adapt_batchnorm(model, loader(target, batch_size=10), "cpu", max_images=40)
    from_40, _ = adapt_batchnorm(model, loader(target[:40], batch_size=10), "cpu", max_images=40)
    assert info["images"] == 40
    for k, v in bn_stats(from_60).items():
        torch.testing.assert_close(v, bn_stats(from_40)[k])


def test_partial_last_batch_is_truncated(model, target):
    _, info = adapt_batchnorm(model, loader(target, batch_size=16), "cpu", max_images=40)
    assert info == {"images": 40, "batches": 3, "bn_layers": 121}  # 16 + 16 + 8


def test_order_invariant_with_equal_batches(model, target):
    a, _ = adapt_batchnorm(model, loader(target[:40]), "cpu", max_images=40)
    b, _ = adapt_batchnorm(model, loader(target[:40].flip(0)), "cpu", max_images=40)
    torch.testing.assert_close(a.features.norm0.running_mean, b.features.norm0.running_mean, rtol=1e-5, atol=1e-6)


def test_statistics_frozen_after_adaptation(model, target):
    adapted, _ = adapt_batchnorm(model, loader(target), "cpu", max_images=40)
    assert not adapted.training and not any(m.training for m in adapted.modules())
    assert not any(p.requires_grad for p in adapted.parameters())
    frozen = bn_stats(adapted)
    with torch.no_grad():
        out1 = adapted(torch.randn(4, 3, 64, 64) * 5)
        adapted(torch.randn(4, 3, 64, 64) - 3)
    for k, v in bn_stats(adapted).items():
        assert torch.equal(v, frozen[k]), f"{k} changed during evaluation"
    # eval output is deterministic and per-sample (no batch dependence)
    x = torch.randn(4, 3, 64, 64)
    with torch.no_grad():
        full = adapted(x)
        single = torch.cat([adapted(x[i:i + 1]) for i in range(4)])
    torch.testing.assert_close(full, single, rtol=1e-4, atol=1e-5)
    assert out1.shape == (4, 3)


def test_adaptation_changes_predictions(model, target):
    adapted, _ = adapt_batchnorm(model, loader(target), "cpu", max_images=40)
    with torch.no_grad():
        assert not torch.allclose(model(target[40:]), adapted(target[40:]))


def test_rejects_labeled_batches(model, target):
    labeled = DataLoader(TensorDataset(target, torch.zeros(len(target), 3)), batch_size=10)
    with pytest.raises(TypeError, match="images only"):
        adapt_batchnorm(model, labeled, "cpu", max_images=20)


def test_rejects_too_few_images(model, target):
    with pytest.raises(ValueError, match="expected 1000"):
        adapt_batchnorm(model, loader(target), "cpu", max_images=1000)


def test_rejects_model_without_batchnorm():
    with pytest.raises(ValueError, match="no BatchNorm"):
        adapt_batchnorm(nn.Sequential(nn.Flatten(), nn.Linear(12, 2)), loader(torch.randn(10, 3, 2, 2)), "cpu", 10)
