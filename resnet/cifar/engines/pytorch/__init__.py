"""
@Author: DAShaikh10
@Description: `resnet.cifar.pytorch.engines` package containing all PyTorch code related to training and evaluating
              Residual Networks (ResNets) on CIFAR-10 and CIFAR-100 datasets.
"""

from .engine import ResNetEngine
from .gpu_engine import TorchResNetEngine
from .tpu_engine import TorchResNetTPUEngine

__all__ = ["ResNetEngine", "TorchResNetEngine", "TorchResNetTPUEngine"]
