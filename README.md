# cv-training-lab

A small, reproducible computer-vision training lab on CIFAR-10: architecture comparison, hyperparameter search, a model registry with versions and checksums, Grad-CAM with a quantitative check, and drift monitoring.

Design note: `docs/design.md`. Raw results: `results/lab-results.json`.

## Results

Train subset 20,000 images, validation 5,000, test 10,000 (official). One run per row, seed 0. A rerun on the same machine gave the same numbers.

| Model | Epochs | Test accuracy | Train time | Parameters | GPU latency per image |
|---|---|---|---|---|---|
| Small CNN, scratch | 10 | 0.706 | 9 s | 0.09 M | 0.014 ms |
| ResNet-18, scratch | 10 | 0.837 | 76 s | 11.2 M | 0.274 ms |
| ResNet-18, ImageNet weights | 10 | 0.904 | 33 s | 11.2 M | 0.095 ms |

Hyperparameter search (small CNN, 5 epochs, 8 random trials): default configuration 0.656 test accuracy, best trial retrained 0.694. The best trial used a learning rate near 0.14 and batch size 64. The default learning rate was too low for 5 epochs.

Registry: five versions were registered. A version becomes production only when it beats the incumbent by 0.5 points on validation. Unit tests cover promotion, archiving and checkpoint tamper detection.

Grad-CAM deletion test (500 correctly classified test images, mask 20% of the image):

| Model | Probability drop, CAM region | Probability drop, random square |
|---|---|---|
| ResNet-18, scratch | 0.41 | 0.27 |
| ResNet-18, ImageNet weights | 0.15 | 0.33 |

For the scratch model the CAM region matters more than a random region, so the explanation is informative. For the pretrained model it does not: the last block has a 2 by 2 map on the 64 px input, which is too coarse to localise.

Drift monitor (ResNet-18 pretrained, 25 held-out batches of 200 images): false alarm rate 0.00 on clean batches. Detection rate 1.00 for noise sigma 0.05, noise sigma 0.15 and 5 by 5 blur, and 0.96 for brightness x1.5. Accuracy under those corruptions: 0.78, 0.30, 0.35 and 0.86. Brightness x1.5 raises the alarm while accuracy stays at 0.86, so the monitor flags distribution change, not only accuracy loss.

## Limits

- One seed and one run per row. Gaps of a point or two are not established.
- Training on 20,000 images, not the full 50,000, so absolute accuracy is below published CIFAR-10 numbers.
- The search is small (8 trials) and uses the small CNN, not the ResNets.
- The pretrained model sees ImageNet-pretrained weights and 64 px input, so its comparison with the scratch model mixes pretraining and resolution.
- Single GPU (RTX 3050 6 GB). No distributed training and no cloud training service were used.
- Monitoring thresholds come from 25 calibration batches. The false alarm rate has wide uncertainty.
- A first deletion test used an unfair random baseline. See `docs/design.md`.

## Run it

```sh
uv venv .venv && uv pip install --python .venv/bin/python -e . pytest pandas pyarrow pillow
curl -L -o data/train.parquet https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/train-00000-of-00001.parquet
curl -L -o data/test.parquet https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/test-00000-of-00001.parquet
.venv/bin/python scripts/prepare_data.py
PYTHONPATH=src .venv/bin/python -m pytest
PYTHONPATH=src .venv/bin/python -m training_lab.experiments data registry results/lab-results.json
```

Needs a CUDA GPU.
