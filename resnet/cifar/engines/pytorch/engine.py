"""
@Author: DAShaikh10
"""

# pylint: disable=too-many-instance-attributes, unused-import

import json
from pathlib import Path

import torch

from torchvision import datasets, transforms

from torch.utils.data import random_split, DataLoader, SubsetRandomSampler

try:
    import torch_xla

    XLA_AVAILABLE = True
except ImportError:
    XLA_AVAILABLE = False

from utils import enums
from resnet import ResNetConfig

from resnet.cifar.models import TorchResNet

from .gpu_engine import TorchResNetEngine
from .tpu_engine import TorchResNetTPUEngine


class ResNetEngine:
    """
    Base PyTorch ResNet engine class. Also serves as the factory for instantiating
    the appropriate engine based on the configuration and available hardware.

    Supports **CPU/GPU** (via `TorchResNetEngine`) and **TPU** (via `TorchResNetTPUEngine`)
    and automatic hardware detection.
    """

    def __init__(self, config: ResNetConfig):
        self._generator = torch.random.manual_seed(config.seed)  # Sets the global random seed.
        self._train_indices = None
        self._val_indices = None

        self.config = config
        self.device = None
        self.model: TorchResNet = None
        self.test_dataloader: DataLoader = None
        self.test_mean = 0.0
        self.test_std = 0.0
        self.test_transforms: transforms.Compose = None
        self.train_dataloader: DataLoader = None
        self.train_mean = 0.0
        self.train_std = 0.0
        self.train_transforms: transforms.Compose = None
        self.val_dataloader: DataLoader = None
        self.val_mean = 0.0
        self.val_std = 0.0
        self.val_transforms: transforms.Compose = None

    @staticmethod
    def create(config: ResNetConfig):
        """
        Factory method to instantiate the appropriate ResNet engine
        based on the specified configuration and available hardware.

        Supports both CPU/GPU (via `TorchResNetEngine`) and TPU (via `TorchResNetTPUEngine`)
        and automatic hardware detection.
        """

        # Explicit device selection and auto-detection logic.
        # TPU Selection is lazily checked, meaning at this stage we only check `torch_xla` is availability.
        # As, checking TPU availability breaks the runtime.
        if (config.device == enums.Device.TPU) or (config.device is None and XLA_AVAILABLE):
            return TorchResNetTPUEngine(config)

        return TorchResNetEngine(config)

    @staticmethod
    def lr_lambda(current_step: int, learning_rate: list[float], milestones: list[int], enable_warmup: bool):
        """
        Learning rate schedule multiplier function.
        Returns a multiplicative factor for the base learning rate.
        """

        if enable_warmup and current_step < 400:
            # Warmup: return ratio of warmup_lr to base_lr
            return learning_rate[0] / learning_rate[-1]
        if current_step < milestones[0]:
            # Normal rate: 1.0x
            return 1.0
        if current_step < milestones[1]:
            # First decay: 0.1x
            return 0.1

        # Second decay: 0.01x
        return 0.01

    def _compute_statistics(self, dataloader: DataLoader):
        """
        Compute the global mean and standard deviation of the dataset for normalization.
        """

        total_pixels = 0
        sum_ = torch.zeros(3)
        sum_sq = torch.zeros(3)

        for images, _ in dataloader:
            # images shape: [Batch, Channels, Height, Width].
            b, _, h, w = images.shape
            pixels_in_batch = b * h * w

            # Sum of pixels and sum of squared pixels per channel.
            sum_ += torch.sum(images, dim=[0, 2, 3])
            sum_sq += torch.sum(images**2, dim=[0, 2, 3])
            total_pixels += pixels_in_batch

        mean = sum_ / total_pixels
        var = (sum_sq / total_pixels) - (mean**2)  # Variance formula: E[X^2] - (E[X])^2

        return mean, torch.sqrt(var)

    def _init_transforms(self, load_test: bool = False):
        """
        Initialize the data transformations for CIFAR-10 or CIFAR-100 dataset.
        This requires calculation of the mean and standard deviation of the dataset for normalization.
        """

        if load_test:
            self.test_transforms = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(self.test_mean, self.test_std),
                ]
            )
            return

        # The paper specifies use of a random crop of 32x32 with padding of 4 px. and random horizontal flip.
        # Also, the image is normalized by per-pixel subtraction of mean and division by standard deviation.
        self.train_transforms = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(self.train_mean, self.train_std),
            ]
        )
        self.val_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(self.val_mean, self.val_std),
            ]
        )

    def _prepare_dataset(self, load_test: bool = False):
        """
        Prepare the CIFAR-10 or CIFAR-100 dataset and
        compute the mean and standard deviation for training, validation, and testing datasets.
        """

        # Fetch variant of CIFAR. The paper does not use CIFAR-100,
        # but we support it for completeness and experimentation.
        dataset_reader = datasets.CIFAR10 if self.config.variant == enums.CIFAR.CIFAR10 else datasets.CIFAR100

        # Only read the test dataset instead of the training dataset as well.
        if load_test:
            test_dataset = dataset_reader(root="./data", train=False, download=True, transform=transforms.ToTensor())
            test_dataloader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False)
            self.test_mean, self.test_std = self._compute_statistics(test_dataloader)
            del test_dataset, test_dataloader  # Free temporary objects used only for stats.
            return

        train_dataset = dataset_reader(root="./data", train=True, download=True, transform=transforms.ToTensor())

        # Calculate indices for training and validation datasets based on the specified `train_val_split` ratio.
        # It handles the case where a separate val. set is not present and hence the training set is split.
        # Also, if val. set is not desired, we set `train_val_split` to 1., to use all data for training.
        train_split_size = int(len(train_dataset) * self.config.train_val_split)
        [self._train_indices, self._val_indices] = random_split(
            train_dataset,
            [
                train_split_size,
                len(train_dataset) - train_split_size,
            ],
            generator=self._generator,
        )

        # Training dataset.
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,  # No need for shuffle.
            sampler=SubsetRandomSampler(self._train_indices.indices),
        )
        self.train_mean, self.train_std = self._compute_statistics(train_dataloader)

        # Validation dataset.
        val_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            sampler=SubsetRandomSampler(self._val_indices.indices),
        )
        self.val_mean, self.val_std = self._compute_statistics(val_dataloader)
        del train_dataset, train_dataloader, val_dataloader  # Free temporary objects used only for stats.

    def save(self):
        """
        Save the trained model to disk.

        The filename encodes the network depth using the **CIFAR ResNet** formula: `6n + 2`,
        where n is `residual_block_depth` _(e.g. n=5 → ResNet-32, n=9 → ResNet-56)_.
        """

        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)

        # Formula as given in research paper.
        depth = 6 * self.config.residual_block_depth + 2
        stem = f"resnet-{depth}_{self.config.variant.value}"

        # Save model weights.
        weights_path = models_dir / f"{stem}.pth"
        torch.save(self.model.state_dict(), weights_path)
        print(f"Model weights saved to {weights_path}")

        # Save ModelConfig so the architecture can be reconstructed on load.
        model_config = {
            "initial_out_planes": self.model.config.initial_out_planes,
            "kernel_size": self.model.config.kernel_size,
            "num_classes": self.model.config.num_classes,
            "padding": self.model.config.padding,
            "residual_block_depth": self.config.residual_block_depth,
            "stride": self.model.config.stride,
        }
        config_path = models_dir / f"{stem}.json"
        config_path.write_text(json.dumps(model_config, indent=2), encoding="utf-8")
        print(f"Model config saved to {config_path}")
