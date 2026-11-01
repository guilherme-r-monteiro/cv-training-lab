from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)
CLASS_COUNT = 10


@dataclass(frozen=True)
class Split:
    images: torch.Tensor
    labels: torch.Tensor


@dataclass(frozen=True)
class Dataset:
    train: Split
    validation: Split
    test: Split


def to_split(images: np.ndarray, labels: np.ndarray, device: torch.device) -> Split:
    tensor = torch.from_numpy(images).permute(0, 3, 1, 2).contiguous().to(device)
    return Split(tensor, torch.from_numpy(labels).long().to(device))


def take(split: Split, indexes: torch.Tensor) -> Split:
    return Split(split.images[indexes], split.labels[indexes])


def load_archive(path: Path, device: torch.device) -> Split:
    archive = np.load(path)
    return to_split(archive["images"], archive["labels"], device)


def load_dataset(root: Path, train_size: int, validation_size: int, seed: int, device: torch.device) -> Dataset:
    full = load_archive(root / "train.npz", device)
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(full.labels), generator=generator).to(device)
    train = take(full, order[:train_size])
    validation = take(full, order[train_size : train_size + validation_size])
    return Dataset(train, validation, load_archive(root / "test.npz", device))


def normalize(images: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(MEAN, device=images.device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=images.device).view(1, 3, 1, 1)
    return (images.float() / 255.0 - mean) / std


def augment(images: torch.Tensor, padding: int, flip: bool, generator: torch.Generator) -> torch.Tensor:
    count, _, height, width = images.shape
    padded = torch.nn.functional.pad(images, (padding,) * 4, mode="reflect")
    offsets = torch.randint(0, 2 * padding + 1, (count, 2), generator=generator, device="cpu")
    crops = torch.stack(
        [padded[i, :, int(offsets[i, 0]) : int(offsets[i, 0]) + height, int(offsets[i, 1]) : int(offsets[i, 1]) + width] for i in range(count)]
    )
    if not flip:
        return crops
    mirrored = torch.rand(count, generator=generator, device="cpu") < 0.5
    return torch.where(mirrored.to(images.device).view(-1, 1, 1, 1), crops.flip(3), crops)
