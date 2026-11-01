import hashlib
import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

import torch
from torch import nn

from training_lab.config import TrainConfig, config_hash

PROMOTION_MARGIN = 0.005


class Stage(StrEnum):
    CANDIDATE = "candidate"
    PRODUCTION = "production"
    ARCHIVED = "archived"


@dataclass(frozen=True)
class ModelVersion:
    name: str
    version: int
    stage: Stage
    config_hash: str
    config: dict[str, object]
    checkpoint_sha256: str
    validation_accuracy: float
    test_accuracy: float


def index_path(root: Path) -> Path:
    return root / "index.json"


def load_versions(root: Path) -> list[ModelVersion]:
    path = index_path(root)
    if not path.exists():
        return []
    return [ModelVersion(**{**row, "stage": Stage(row["stage"])}) for row in json.loads(path.read_text())]


def save_versions(root: Path, versions: list[ModelVersion]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    index_path(root).write_text(json.dumps([asdict(v) for v in versions], indent=2))


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def production_version(versions: list[ModelVersion], name: str) -> ModelVersion | None:
    return next((v for v in versions if v.name == name and v.stage is Stage.PRODUCTION), None)


def decide_stage(incumbent: ModelVersion | None, validation_accuracy: float) -> Stage:
    if incumbent is None or validation_accuracy >= incumbent.validation_accuracy + PROMOTION_MARGIN:
        return Stage.PRODUCTION
    return Stage.CANDIDATE


def register(
    root: Path, name: str, model: nn.Module, config: TrainConfig, validation_accuracy: float, test_accuracy: float
) -> ModelVersion:
    versions = load_versions(root)
    number = 1 + max((v.version for v in versions if v.name == name), default=0)
    checkpoint = root / name / f"v{number}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint)
    stage = decide_stage(production_version(versions, name), validation_accuracy)
    created = ModelVersion(name, number, stage, config_hash(config), asdict(config), file_sha256(checkpoint), validation_accuracy, test_accuracy)
    demoted = [
        ModelVersion(**{**asdict(v), "stage": Stage.ARCHIVED}) if stage is Stage.PRODUCTION and v.name == name and v.stage is Stage.PRODUCTION else v
        for v in versions
    ]
    save_versions(root, demoted + [created])
    return created


def verify(root: Path, version: ModelVersion) -> bool:
    return file_sha256(root / version.name / f"v{version.version}.pt") == version.checkpoint_sha256
