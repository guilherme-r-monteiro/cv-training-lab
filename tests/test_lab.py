import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from training_lab.config import Architecture, TrainConfig, config_hash
from training_lab.explain import top_fraction_mask
from training_lab.monitor import Corruption, batch_statistics, corrupt
from training_lab.registry import Stage, decide_stage, load_versions, register, verify
from training_lab.search import BATCH_SIZES, sample_config

CONFIG = TrainConfig(architecture=Architecture.SMALL_CNN)


def test_config_hash_is_stable_and_sensitive() -> None:
    assert config_hash(CONFIG) == config_hash(TrainConfig(architecture=Architecture.SMALL_CNN))
    assert config_hash(CONFIG) != config_hash(TrainConfig(architecture=Architecture.SMALL_CNN, seed=1))


def test_first_version_goes_to_production(tmp_path: Path) -> None:
    version = register(tmp_path, "m", nn.Linear(2, 2), CONFIG, 0.8, 0.79)
    assert version.stage is Stage.PRODUCTION and verify(tmp_path, version)


def test_small_gain_stays_candidate_and_large_gain_replaces(tmp_path: Path) -> None:
    register(tmp_path, "m", nn.Linear(2, 2), CONFIG, 0.80, 0.8)
    small = register(tmp_path, "m", nn.Linear(2, 2), CONFIG, 0.802, 0.8)
    large = register(tmp_path, "m", nn.Linear(2, 2), CONFIG, 0.85, 0.8)
    stages = {v.version: v.stage for v in load_versions(tmp_path)}
    assert small.stage is Stage.CANDIDATE and large.stage is Stage.PRODUCTION
    assert stages == {1: Stage.ARCHIVED, 2: Stage.CANDIDATE, 3: Stage.PRODUCTION}


def test_tampered_checkpoint_fails_verification(tmp_path: Path) -> None:
    version = register(tmp_path, "m", nn.Linear(2, 2), CONFIG, 0.8, 0.8)
    (tmp_path / "m" / "v1.pt").write_bytes(b"x")
    assert not verify(tmp_path, version)


def test_decide_stage_without_incumbent() -> None:
    assert decide_stage(None, 0.1) is Stage.PRODUCTION


def test_sampled_config_stays_in_range() -> None:
    rng = random.Random(0)
    for _ in range(50):
        sampled = sample_config(CONFIG, rng)
        assert 0.01 <= sampled.learning_rate <= 0.2 and sampled.batch_size in BATCH_SIZES


def test_top_fraction_mask_selects_highest_scores() -> None:
    scores = torch.arange(100, dtype=torch.float32).view(1, 10, 10)
    mask = top_fraction_mask(scores, 0.2)
    assert int(mask.sum()) == 20 and bool(mask.flatten()[-1]) and not bool(mask.flatten()[0])


def test_corruptions_keep_shape_and_change_pixels() -> None:
    images = torch.randint(0, 255, (4, 3, 32, 32), dtype=torch.uint8)
    generator = torch.Generator().manual_seed(0)
    for corruption in Corruption:
        changed = corrupt(images, corruption, generator)
        assert changed.shape == images.shape and not torch.equal(changed, images)


def test_batch_statistics_is_larger_for_shifted_scores() -> None:
    rng = np.random.default_rng(0)
    reference = rng.normal(0.9, 0.05, 2000)
    same = batch_statistics(rng.normal(0.9, 0.05, 400), reference)
    shifted = batch_statistics(rng.normal(0.7, 0.05, 400), reference)
    assert min(shifted) > max(same)
