"""
@Author: DAShaikh10
@Description: `resnet.cifar.engines` package containing all code related to training and evaluating
              Residual Networks (ResNets) on CIFAR-10 and CIFAR-100 datasets.
"""

from .pytorch.engine import ResNetEngine
from .pytorch.gpu_engine import TorchResNetEngine
from .pytorch.tpu_engine import TorchResNetTPUEngine

__all__ = ["ResNetEngine", "TorchResNetEngine", "TorchResNetTPUEngine"]
