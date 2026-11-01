import hashlib
import json
from dataclasses import asdict, dataclass
from enum import StrEnum


class Architecture(StrEnum):
    SMALL_CNN = "small_cnn"
    RESNET18_SCRATCH = "resnet18_scratch"
    RESNET18_PRETRAINED = "resnet18_pretrained"


@dataclass(frozen=True)
class TrainConfig:
    architecture: Architecture
    learning_rate: float = 0.05
    weight_decay: float = 5e-4
    epochs: int = 10
    batch_size: int = 128
    seed: int = 0
    train_size: int = 20000
    validation_size: int = 5000
    crop_padding: int = 4
    flip: bool = True


def config_hash(config: TrainConfig) -> str:
    payload = json.dumps(asdict(config), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:12]
