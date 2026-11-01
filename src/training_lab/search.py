import math
import random
from dataclasses import dataclass, replace

from training_lab.config import Architecture, TrainConfig
from training_lab.data import Dataset
from training_lab.train import train_model

LEARNING_RATE_RANGE = (0.01, 0.2)
WEIGHT_DECAY_RANGE = (1e-5, 1e-2)
BATCH_SIZES = (64, 128, 256)


@dataclass(frozen=True)
class Trial:
    config: TrainConfig
    validation_accuracy: float


def log_uniform(rng: random.Random, low: float, high: float) -> float:
    return math.exp(rng.uniform(math.log(low), math.log(high)))


def sample_config(base: TrainConfig, rng: random.Random) -> TrainConfig:
    return replace(
        base,
        learning_rate=log_uniform(rng, *LEARNING_RATE_RANGE),
        weight_decay=log_uniform(rng, *WEIGHT_DECAY_RANGE),
        batch_size=rng.choice(BATCH_SIZES),
    )


def random_search(base: TrainConfig, dataset: Dataset, trials: int, seed: int) -> list[Trial]:
    rng = random.Random(seed)
    results = []
    for _ in range(trials):
        config = sample_config(base, rng)
        results.append(Trial(config, train_model(config, dataset).validation_accuracy))
    return sorted(results, key=lambda trial: -trial.validation_accuracy)


def default_config(architecture: Architecture, epochs: int) -> TrainConfig:
    return TrainConfig(architecture=architecture, epochs=epochs)
