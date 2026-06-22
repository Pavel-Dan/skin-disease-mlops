from __future__ import annotations

import os

import torch
import torch.nn as nn
from torchvision import models, transforms

from src.utils.config import CLASSES, IMG_SIZE, MODEL_ARCH


def build_model(
    num_classes: int | None = None,
    pretrained: bool = True,
    arch: str | None = None,
) -> nn.Module:
    n = num_classes or len(CLASSES)
    arch_name = (arch or MODEL_ARCH).lower()

    if arch_name in {"efficientnet_b3", "efficientnet-b3", "efficientnetb3"}:
        weights = models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b3(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, n)
        return model

    if arch_name in {"efficientnet_b0", "efficientnet-b0"}:
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, n)
        return model

    if arch_name in {"resnet101", "resnet-101"}:
        weights = models.ResNet101_Weights.DEFAULT if pretrained else None
        model = models.resnet101(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, n)
        return model

    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, n)
    return model


def get_model_transforms(train: bool = False):
    if train:
        return transforms.Compose(
            [
                transforms.Resize(IMG_SIZE),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
