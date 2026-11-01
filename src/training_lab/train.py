import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from training_lab.config import TrainConfig
from training_lab.data import Dataset, Split, augment, normalize
from training_lab.models import build_model, parameter_count

EVAL_BATCH = 500
LATENCY_BATCH = 64
LATENCY_REPEATS = 20


@dataclass(frozen=True)
class TrainResult:
    model: nn.Module
    validation_accuracy: float
    test_accuracy: float
    train_seconds: float
    parameters: int
    latency_ms_per_image: float


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict_logits(model: nn.Module, split: Split) -> torch.Tensor:
    model.eval()
    chunks = []
    for start in range(0, len(split.labels), EVAL_BATCH):
        batch = normalize(split.images[start : start + EVAL_BATCH]).contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.float16):
            chunks.append(model(batch).float())
    return torch.cat(chunks)


def accuracy(model: nn.Module, split: Split) -> float:
    return float((predict_logits(model, split).argmax(1) == split.labels).float().mean())


@torch.no_grad()
def latency_ms(model: nn.Module, split: Split) -> float:
    model.eval()
    batch = normalize(split.images[:LATENCY_BATCH]).contiguous(memory_format=torch.channels_last)
    for _ in range(3):
        model(batch)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(LATENCY_REPEATS):
        model(batch)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / (LATENCY_REPEATS * LATENCY_BATCH) * 1000


def train_model(config: TrainConfig, dataset: Dataset) -> TrainResult:
    seed_everything(config.seed)
    device = dataset.train.images.device
    model = build_model(config.architecture).to(device).to(memory_format=torch.channels_last)
    steps = config.epochs * (len(dataset.train.labels) // config.batch_size)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9, weight_decay=config.weight_decay, nesterov=True)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config.learning_rate, total_steps=steps)
    scaler = torch.amp.GradScaler()
    generator = torch.Generator().manual_seed(config.seed)
    started = time.perf_counter()
    for _ in range(config.epochs):
        model.train()
        order = torch.randperm(len(dataset.train.labels), generator=generator).to(device)
        for start in range(0, len(order) - config.batch_size + 1, config.batch_size):
            indexes = order[start : start + config.batch_size]
            images = augment(dataset.train.images[indexes], config.crop_padding, config.flip, generator)
            batch = normalize(images).contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = nn.functional.cross_entropy(model(batch), dataset.train.labels[indexes])
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            schedule.step()
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    return TrainResult(
        model,
        accuracy(model, dataset.validation),
        accuracy(model, dataset.test),
        seconds,
        parameter_count(model),
        latency_ms(model, dataset.test),
    )
