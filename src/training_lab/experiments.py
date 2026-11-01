import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import torch

from training_lab.config import Architecture, TrainConfig
from training_lab.data import load_dataset
from training_lab.explain import deletion_test
from training_lab.monitor import monitor
from training_lab.registry import register
from training_lab.search import default_config, random_search
from training_lab.train import train_model

COMPARISON_EPOCHS = 10
SEARCH_EPOCHS = 5
SEARCH_TRIALS = 8
PRETRAINED_LEARNING_RATE = 0.02
SEARCH_SEED = 123
CALIBRATION_SIZE = 5000


def summary(config: TrainConfig, result) -> dict[str, object]:
    return {
        "config": asdict(config),
        "validation_accuracy": result.validation_accuracy,
        "test_accuracy": result.test_accuracy,
        "train_seconds": result.train_seconds,
        "parameters": result.parameters,
        "latency_ms_per_image": result.latency_ms_per_image,
    }


def comparison_configs() -> list[TrainConfig]:
    scratch = default_config(Architecture.SMALL_CNN, COMPARISON_EPOCHS)
    resnet = default_config(Architecture.RESNET18_SCRATCH, COMPARISON_EPOCHS)
    pretrained = replace(
        default_config(Architecture.RESNET18_PRETRAINED, COMPARISON_EPOCHS), learning_rate=PRETRAINED_LEARNING_RATE
    )
    return [scratch, resnet, pretrained]


def final_stage(model: torch.nn.Module) -> torch.nn.Module:
    return model.layer4 if hasattr(model, "layer4") else model.inner.layer4


def main(data_root: Path, registry_root: Path, output: Path) -> None:
    device = torch.device("cuda")
    first = comparison_configs()[0]
    dataset = load_dataset(data_root, first.train_size, first.validation_size, first.seed, device)
    report: dict[str, object] = {"comparison": [], "search": {}, "registry": [], "explain": {}, "monitor": {}}
    trained = {}
    for config in comparison_configs():
        result = train_model(config, dataset)
        trained[config.architecture] = result
        report["comparison"].append(summary(config, result))
        version = register(registry_root, config.architecture.value, result.model, config, result.validation_accuracy, result.test_accuracy)
        report["registry"].append({"name": version.name, "version": version.version, "stage": version.stage.value})
        sys.stdout.write(f"{config.architecture.value} val={result.validation_accuracy:.4f} test={result.test_accuracy:.4f} t={result.train_seconds:.0f}s\n")
        sys.stdout.flush()
    base = default_config(Architecture.SMALL_CNN, SEARCH_EPOCHS)
    baseline = train_model(base, dataset)
    trials = random_search(base, dataset, SEARCH_TRIALS, SEARCH_SEED)
    best = train_model(trials[0].config, dataset)
    report["search"] = {
        "default": summary(base, baseline),
        "trials": [{"config": asdict(t.config), "validation_accuracy": t.validation_accuracy} for t in trials],
        "best_retrained": summary(trials[0].config, best),
    }
    for result, config in ((baseline, base), (best, trials[0].config)):
        version = register(registry_root, "small_cnn_search", result.model, config, result.validation_accuracy, result.test_accuracy)
        report["registry"].append({"name": version.name, "version": version.version, "stage": version.stage.value})
    strongest = max(trained, key=lambda arch: trained[arch].validation_accuracy)
    report["explain"] = {
        arch.value: asdict(deletion_test(result.model, final_stage(result.model), dataset.test, 0))
        for arch, result in trained.items()
        if arch is not Architecture.SMALL_CNN
    }
    model = trained[strongest].model
    held_out = type(dataset.test)(dataset.test.images[CALIBRATION_SIZE:], dataset.test.labels[CALIBRATION_SIZE:])
    calibration = type(dataset.test)(dataset.test.images[:CALIBRATION_SIZE], dataset.test.labels[:CALIBRATION_SIZE])
    drift = monitor(model, dataset.validation, calibration, held_out, 0)
    report["monitor"] = {"model": strongest.value, **asdict(drift)}
    output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
