"""
@Author: DAShaikh10
@Description: `resnet.cifar.models` package containing all code related to Residual Network (ResNet) models
              for CIFAR-10 and CIFAR-100 datasets.
"""

from .config import ModelConfig
from .pt import TorchResNet


__all__ = ["ModelConfig", "TorchResNet"]
