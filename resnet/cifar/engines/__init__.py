"""
@Author: DAShaikh10
@Description: `resnet.cifar.engines` package containing all code related to training and evaluating
              Residual Networks (ResNets) on CIFAR-10 and CIFAR-100 datasets.
"""

from .pt_engine import TorchResNetEngine

__all__ = ["TorchResNetEngine"]
