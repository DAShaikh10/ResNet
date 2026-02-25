"""
@Author: DAShaikh10
@Description: `resnet.cifar.models` package containing all code related to Residual Network (ResNet) model config.
"""

from dataclasses import dataclass

import torch


@dataclass
class ModelConfig:
    """
    Configuration data-class for Residual Network (ResNet) model.
    """

    device: torch.device
    initial_out_planes: int
    kernel_size: int
    num_classes: int
    padding: int
    residual_block_depth: int
    stride: int
