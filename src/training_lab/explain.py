from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional

from training_lab.data import Split, normalize

MASK_FRACTION = 0.2
SAMPLE_COUNT = 500


@dataclass(frozen=True)
class DeletionResult:
    samples: int
    probability_drop_cam: float
    probability_drop_patch: float
    accuracy_after_cam: float
    accuracy_after_patch: float


def grad_cam(model: nn.Module, layer: nn.Module, images: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    stored: dict[str, torch.Tensor] = {}
    forward = layer.register_forward_hook(lambda _m, _i, out: stored.__setitem__("act", out))
    backward = layer.register_full_backward_hook(lambda _m, _gi, go: stored.__setitem__("grad", go[0]))
    model.eval()
    scores = model(images)
    model.zero_grad()
    scores.gather(1, targets.view(-1, 1)).sum().backward()
    forward.remove()
    backward.remove()
    weights = stored["grad"].mean(dim=(2, 3), keepdim=True)
    cam = functional.relu((weights * stored["act"]).sum(1, keepdim=True))
    cam = functional.interpolate(cam, size=images.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
    flat = cam.flatten(1)
    return (flat / (flat.amax(1, keepdim=True) + 1e-9)).view_as(cam)


def top_fraction_mask(scores: torch.Tensor, fraction: float) -> torch.Tensor:
    flat = scores.flatten(1)
    count = int(flat.shape[1] * fraction)
    indexes = flat.topk(count, dim=1).indices
    mask = torch.zeros_like(flat, dtype=torch.bool)
    mask.scatter_(1, indexes, True)
    return mask.view_as(scores)


def random_patch_mask(count: int, height: int, width: int, fraction: float, generator: torch.Generator) -> torch.Tensor:
    side_h, side_w = round(height * fraction**0.5), round(width * fraction**0.5)
    tops = torch.randint(0, height - side_h + 1, (count,), generator=generator)
    lefts = torch.randint(0, width - side_w + 1, (count,), generator=generator)
    rows = torch.arange(height).view(1, -1, 1)
    cols = torch.arange(width).view(1, 1, -1)
    inside_rows = (rows >= tops.view(-1, 1, 1)) & (rows < (tops + side_h).view(-1, 1, 1))
    inside_cols = (cols >= lefts.view(-1, 1, 1)) & (cols < (lefts + side_w).view(-1, 1, 1))
    return inside_rows & inside_cols


@torch.no_grad()
def class_probability(model: nn.Module, images: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probabilities = functional.softmax(model(images), dim=1)
    return probabilities.gather(1, targets.view(-1, 1)).squeeze(1)


def deletion_test(model: nn.Module, layer: nn.Module, test: Split, seed: int) -> DeletionResult:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    images = normalize(test.images[:SAMPLE_COUNT * 4])
    model.eval()
    with torch.no_grad():
        correct = model(images).argmax(1) == test.labels[: len(images)]
    images, targets = images[correct][:SAMPLE_COUNT], test.labels[: len(correct)][correct][:SAMPLE_COUNT]
    cam = grad_cam(model, layer, images.clone().requires_grad_(False), targets)
    patch = random_patch_mask(len(targets), cam.shape[1], cam.shape[2], MASK_FRACTION, generator).to(cam.device)
    baseline = class_probability(model, images, targets)

    def after_masking(mask: torch.Tensor) -> tuple[float, float]:
        mask = mask.unsqueeze(1)
        masked = torch.where(mask, torch.zeros_like(images), images)
        probability = class_probability(model, masked, targets)
        with torch.no_grad():
            kept = float((model(masked).argmax(1) == targets).float().mean())
        return float((baseline - probability).mean()), kept

    cam_drop, cam_accuracy = after_masking(top_fraction_mask(cam, MASK_FRACTION))
    patch_drop, patch_accuracy = after_masking(patch)
    return DeletionResult(len(targets), cam_drop, patch_drop, cam_accuracy, patch_accuracy)
