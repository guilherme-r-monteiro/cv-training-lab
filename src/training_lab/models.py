import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from training_lab.config import Architecture
from training_lab.data import CLASS_COUNT

PRETRAINED_INPUT_SIZE = 64


def small_cnn() -> nn.Module:
    def block(inputs: int, outputs: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(inputs, outputs, 3, padding=1, bias=False),
            nn.BatchNorm2d(outputs),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    return nn.Sequential(block(3, 32), block(32, 64), block(64, 128), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, CLASS_COUNT))


def resnet18_for_small_images(pretrained: bool) -> nn.Module:
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    if not pretrained:
        model.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, CLASS_COUNT)
    return model


class Upsampled(nn.Module):
    def __init__(self, inner: nn.Module, size: int) -> None:
        super().__init__()
        self.inner = inner
        self.size = size

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.inner(nn.functional.interpolate(images, size=self.size, mode="bilinear", align_corners=False))


def build_model(architecture: Architecture) -> nn.Module:
    if architecture is Architecture.SMALL_CNN:
        return small_cnn()
    if architecture is Architecture.RESNET18_SCRATCH:
        return resnet18_for_small_images(pretrained=False)
    return Upsampled(resnet18_for_small_images(pretrained=True), PRETRAINED_INPUT_SIZE)


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
