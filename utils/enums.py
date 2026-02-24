"""
@Author: DAShaikh10
"""

from enum import Enum


class IgnoreCaseEnum(str, Enum):
    """
    Base enum class that provides case-insensitive lookup.
    """

    @classmethod
    def _missing_(cls, value):
        """
        Handle case-insensitive lookup.
        """

        if isinstance(value, str):
            for member in cls:
                if member.value.lower() == value.lower():
                    return member

        return None


# TODO: Possibly add more activation functions like LeakyReLU, ELU, etc.
class ActivationFunction(IgnoreCaseEnum):
    """
    Enum for activation functions.
    """

    RELU = "ReLU"


class CIFAR(IgnoreCaseEnum):
    """
    Enum for CIFAR datasets.
    """

    CIFAR10 = "CIFAR-10"
    CIFAR100 = "CIFAR-100"


class Device(IgnoreCaseEnum):
    """
    Enum for devices.
    """

    CPU = "CPU"
    GPU = "GPU"
    TPU = "TPU"


class Framework(IgnoreCaseEnum):
    """
    Enum for deep learning frameworks.
    """

    PYTORCH = "PyTorch"
    TENSORFLOW = "TensorFlow"


# TODO: Possibly add more optimizers like Adam, RMSprop, etc.
class Optimizer(IgnoreCaseEnum):
    """
    Enum for optimizers.
    """

    SGD = "SGD"
    NAGSGD = "NAGSGD"


# TODO: Possibly add more weight initialization methods like Xavier, etc.
class WeightInitialization(IgnoreCaseEnum):
    """
    Enum for weight initialization methods.
    """

    HE = "He"
