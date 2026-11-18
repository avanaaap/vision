"""AdaBN: re-estimate BatchNorm statistics on unlabeled target images.

Li et al., "Revisiting Batch Normalization For Practical Domain Adaptation"
(2016). Every learned weight (conv/linear weights and BN affine gamma/beta) is
kept fixed; only the BN running means/variances are recomputed on target data.

Procedure:
1. Copy the model so the source checkpoint is untouched.
2. Reset each BN layer's running mean/var and set momentum=None, which makes
   PyTorch keep a cumulative (equal-weight) average over all adaptation
   batches instead of an exponential moving average.
3. Put the network in eval mode, then only BN layers in train mode, and run
   forward passes under torch.no_grad() on the target images. No loss, no
   gradients, no optimizer, and no labels: the loader is wrapped so it yields
   images only.
4. Switch everything back to eval mode, which freezes the new statistics for
   evaluation, and verify that no learned parameter changed.
"""

import copy

import torch
from torch import nn
from torch.utils.data import Dataset


class ImagesOnly(Dataset):
    """Wrap a labeled dataset so adaptation code can never see labels."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return item[0] if isinstance(item, (tuple, list)) else item


def _bn_layers(model):
    return [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]


@torch.no_grad()
def adapt_batchnorm(model, image_loader, device, max_images=1000):
    """Return (adapted_model, info). ``image_loader`` must yield image tensors only."""
    adapted = copy.deepcopy(model).to(device)
    learned_before = {name: p.detach().clone() for name, p in adapted.named_parameters()}
    layers = _bn_layers(adapted)
    if not layers:
        raise ValueError("Model has no BatchNorm layers; AdaBN does nothing")
    for layer in layers:
        layer.reset_running_stats()
        layer.momentum = None  # cumulative average over adaptation batches

    adapted.eval()
    for layer in layers:
        layer.train()

    seen, batches = 0, 0
    for images in image_loader:
        if isinstance(images, (tuple, list)):
            raise TypeError("AdaBN loader must yield images only; wrap the dataset with ImagesOnly")
        images = images[: max_images - seen]
        if len(images) == 0:
            break
        adapted(images.to(device))
        seen += len(images)
        batches += 1
        if seen >= max_images:
            break
    if seen < max_images:
        raise ValueError(f"AdaBN expected {max_images} target images but the loader provided {seen}")

    # Freeze: eval mode uses the stored running statistics and stops updating them.
    adapted.eval()
    for p in adapted.parameters():
        p.requires_grad_(False)
    for name, p in adapted.named_parameters():
        if not torch.equal(p, learned_before[name].to(p.device)):
            raise RuntimeError(f"AdaBN changed learned parameter {name}")
    return adapted, {"images": seen, "batches": batches, "bn_layers": len(layers)}
