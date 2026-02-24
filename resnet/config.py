"""
@Author: DAShaikh10
"""

# pylint: disable=too-many-instance-attributes

from dataclasses import dataclass

from utils import enums


@dataclass
class ResNetConfig:
    """
    Configuration data-class for Residual Network (ResNet).
    """

    # Required fields.
    batch_size: int
    device: None | enums.Device
    framework: enums.Framework
    residual_block_depth: int
    seed: int
    variant: enums.CIFAR

    # Training-only fields (not needed for inference).
    activation: enums.ActivationFunction = enums.ActivationFunction.RELU
    iterations: int = 0
    learning_rate: list[float] = None
    optimizer: enums.Optimizer = enums.Optimizer.SGD
    train_val_split: float = 0.8
    weight_initialization: enums.WeightInitialization = enums.WeightInitialization.HE
    warmup_threshold: float = 0.0
