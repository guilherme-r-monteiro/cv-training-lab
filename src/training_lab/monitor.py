from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import torch
from scipy.stats import ks_2samp
from torch import nn
from torch.nn import functional

from training_lab.data import Split, normalize
from training_lab.train import predict_logits

BATCH_SIZE = 200
FALSE_ALARM_QUANTILE = 0.95


class Corruption(StrEnum):
    NOISE_LOW = "noise_sigma_0.05"
    NOISE_HIGH = "noise_sigma_0.15"
    BLUR = "blur_k5"
    BRIGHTNESS = "brightness_x1.5"


@dataclass(frozen=True)
class DriftRow:
    corruption: str
    accuracy: float
    detection_rate: float


@dataclass(frozen=True)
class MonitorReport:
    threshold: float
    false_alarm_rate: float
    rows: list[DriftRow]


def corrupt(images: torch.Tensor, corruption: Corruption, generator: torch.Generator) -> torch.Tensor:
    floats = images.float() / 255.0
    if corruption is Corruption.NOISE_LOW or corruption is Corruption.NOISE_HIGH:
        sigma = 0.05 if corruption is Corruption.NOISE_LOW else 0.15
        noise = torch.randn(floats.shape, generator=generator).to(floats.device) * sigma
        floats = floats + noise
    elif corruption is Corruption.BLUR:
        kernel = torch.ones(3, 1, 5, 5, device=floats.device) / 25
        floats = functional.conv2d(functional.pad(floats, (2, 2, 2, 2), mode="reflect"), kernel, groups=3)
    else:
        floats = floats * 1.5
    return (floats.clamp(0, 1) * 255).to(torch.uint8)


def confidence(model: nn.Module, split: Split) -> np.ndarray:
    return functional.softmax(predict_logits(model, split), dim=1).amax(1).cpu().numpy()


def batch_statistics(scores: np.ndarray, reference: np.ndarray) -> list[float]:
    return [ks_2samp(reference, scores[i : i + BATCH_SIZE]).statistic for i in range(0, len(scores) - BATCH_SIZE + 1, BATCH_SIZE)]


def monitor(model: nn.Module, reference: Split, calibration: Split, held_out: Split, seed: int) -> MonitorReport:
    reference_scores = confidence(model, reference)
    threshold = float(np.quantile(batch_statistics(confidence(model, calibration), reference_scores), FALSE_ALARM_QUANTILE))
    clean = batch_statistics(confidence(model, held_out), reference_scores)
    generator = torch.Generator().manual_seed(seed)
    rows = []
    for corruption in Corruption:
        corrupted = Split(corrupt(held_out.images, corruption, generator), held_out.labels)
        statistics = batch_statistics(confidence(model, corrupted), reference_scores)
        accuracy = float((predict_logits(model, corrupted).argmax(1) == held_out.labels).float().mean())
        rows.append(DriftRow(corruption.value, accuracy, float(np.mean(np.array(statistics) > threshold))))
    return MonitorReport(threshold, float(np.mean(np.array(clean) > threshold)), rows)
