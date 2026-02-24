"""
@Author: DAShaikh10
"""

from dataclasses import dataclass

from utils import enums


@dataclass
class ResNetConfig:
    """
    Configuration data-class for Residual Network (ResNet).
    """

    activation: enums.ActivationFunction
    batch_size: int
    device: None | enums.Device
    framework: enums.Framework
    iterations: int
    learning_rate: list[float]
    optimizer: enums.Optimizer
    residual_block_depth: int
    seed: int
    train_val_split: float
    variant: enums.CIFAR
    weight_initialization: enums.WeightInitialization
    warmup_threshold: float
