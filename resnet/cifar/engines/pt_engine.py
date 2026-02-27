"""
@Author: DAShaikh10
@Description: PyTorch engine for training and evaluating Residual Networks (ResNets)
              on CIFAR-10 and CIFAR-100 datasets.
"""

# pylint: disable=too-many-instance-attributes, too-many-locals

import json
from pathlib import Path
from functools import partial

import torch
import torchinfo

try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.distributed.xla_multiprocessing as xmp

    XLA_AVAILABLE = True
    # XLA_AVAILABLE only means torch_xla imported; TPU_AVAILABLE confirms real TPU hardware is present.
    TPU_AVAILABLE = xm.xla_device_hw(torch_xla.device()) == "TPU"
except ImportError:
    XLA_AVAILABLE = False
    TPU_AVAILABLE = False

from tqdm import tqdm
from torch import nn, optim, distributed
from torch.utils.data import DataLoader, random_split, Subset, SubsetRandomSampler
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms

from utils import enums
from resnet import ResNetConfig
from resnet.cifar.models import ModelConfig, TorchResNet


class TorchResNetEngine:
    """
    PyTorch engine for CIFAR ResNet.
    """

    def __init__(self, config: ResNetConfig):
        self._generator = torch.random.manual_seed(config.seed)  # Sets the global random seed.
        self._train_indices = None
        self._val_indices = None

        self.config = config
        self.device = None

        # Auto-detect and set device (GPU if available, else CPU).
        self._setup_device()
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

    @property
    def _is_tpu(self) -> bool:
        """
        Returns True if the current device is a real XLA/TPU device.
        """

        return TPU_AVAILABLE and str(self.device).startswith("xla")

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

    @staticmethod
    def spawn(config: ResNetConfig, nprocs: int = 8) -> None:
        """
        Launch multi-core TPU training using **xmp.spawn**.

        Each TPU core runs an independent process. `DistributedSampler` inside
        `init_dataloader` ensures every core sees a disjoint shard of the data.
        Only the rank-0 process saves the model after training.

        Args:
            config: ResNetConfig - Training configuration.
            nprocs: int - Number of TPU cores to use. _(Default: 8 for a v3/v4 TPU)_
        """

        if not TPU_AVAILABLE:
            raise RuntimeError("spawn() requires torch_xla to be installed and real TPU hardware to be present.")

        def _train_fn(_: int, config: ResNetConfig) -> None:
            engine = TorchResNetEngine(config)
            engine.init_dataloader()
            engine.fit()
            # Only rank-0 saves to avoid multiple processes writing the same file.
            if engine._xla_ordinal() == 0:
                engine.save()

        xmp.spawn(_train_fn, args=(config,), nprocs=nprocs)

    def _xla_world_size(self) -> int:
        """
        Returns the number of XLA devices (TPU cores) in the current process group.
        the standard torch.distributed world size is the correct replacement.
        """

        return distributed.get_world_size() if distributed.is_initialized() else 1


    def _xla_ordinal(self) -> int:
        """
        Returns the ordinal (rank) of the current XLA device.
        """

        return distributed.get_rank() if distributed.is_initialized() else 0

    def _setup_device(self):
        """
        Setup the device for training. Auto-detects TPU > GPU > CPU when no device is specified.
        """

        if self.config.device is None:
            # Auto-detect device: only route to TPU if real hardware is confirmed.
            if TPU_AVAILABLE:
                self.device = torch_xla.device()
                print(f"TPU detected: {self.device}")
                print("Using TPU for training.")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
                print(f"GPU detected: {torch.cuda.get_device_name(0)}")
                print("Using GPU for training.")
            else:
                self.device = torch.device("cpu")
                print("No GPU detected. Using CPU for training.")
        else:
            if self.config.device == enums.Device.TPU:
                if TPU_AVAILABLE:
                    self.device = torch_xla.device()
                    print(f"TPU detected: {self.device}")
                    print("Using TPU for training.")
                    return

                # torch_xla is either not installed or no real TPU hardware found — fall through to GPU.
                reason = "`torch_xla` is not installed" if not XLA_AVAILABLE else "no TPU hardware found"
                print(f"Warning: TPU requested but {reason}. Falling back to GPU/CPU.")
                if torch.cuda.is_available():
                    self.device = torch.device("cuda")
                    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
                    print("Using GPU for training.")
                    return
            if self.config.device == enums.Device.GPU:
                if torch.cuda.is_available():
                    self.device = torch.device("cuda")
                    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
                    print("Using GPU for training.")
                    return

                print("Warning: GPU requested but not available. Falling back to CPU.")

            self.device = torch.device("cpu")
            print("Using CPU for training.")

    def _compute_statistics(self, dataloader: DataLoader):
        """
        Compute the global mean and standard deviation of the dataset for normalization.
        """

        sum_ = torch.zeros(3)
        sum_sq = torch.zeros(3)
        total_pixels = 0

        for images, _ in dataloader:
            # images shape: [Batch, Channels, Height, Width].
            b, _, h, w = images.shape
            pixels_in_batch = b * h * w

            # Sum of pixels and sum of squared pixels per channel.
            sum_ += torch.sum(images, dim=[0, 2, 3])
            sum_sq += torch.sum(images**2, dim=[0, 2, 3])
            total_pixels += pixels_in_batch

        mean = sum_ / total_pixels
        # Variance formula: E[X^2] - (E[X])^2
        var = (sum_sq / total_pixels) - (mean**2)
        std = torch.sqrt(var)

        return mean, std

    def _init_model(self):
        """
        Initialize the ResNet model based on the specified configuration.
        """

        model_config = ModelConfig(
            device=self.device,
            initial_out_planes=16,
            kernel_size=3,
            num_classes=10 if self.config.variant == enums.CIFAR.CIFAR10 else 100,
            residual_block_depth=self.config.residual_block_depth,
            stride=2,
            padding=1,
        )
        self.model = TorchResNet(model_config).to(self.device)
        print(f"Model initialized and moved to {self.device}.")

        # torchinfo creates dummy CPU inputs internally and runs a forward pass through the model.
        # This will cause a device mismatch if the model is on XLA (TPU), so skip it in that case.
        if not self._is_tpu:
            torchinfo.summary(self.model, input_size=(self.config.batch_size, 3, 32, 32), device=self.device)
        print("\n")

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
        # Also, the image is normalized.
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

    def evaluate(self, dataloader: DataLoader, criterion=None):
        """
        Evaluate model on given dataloader. Returns accuracy and optionally loss.
        """

        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0.0

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs: torch.Tensor = self.model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                if criterion is not None:
                    total_loss += criterion(outputs, targets).item()

        accuracy = correct / total
        avg_loss = total_loss / len(dataloader) if criterion is not None else None

        self.model.train()  # Switch back to train mode.

        if criterion is not None:
            return accuracy, avg_loss
        return accuracy

    def init_dataloader(self, load_test: bool = False):
        """
        Initialize the data loader for **CIFAR-10** or **CIFAR-100** dataset.

        Args:
            load_test: bool - Whether to load the test dataset. _(Default: False)_
        """

        print("Downloading / Loading the dataset and computing statistics. This may take a while...")

        self._prepare_dataset(load_test=load_test)
        self._init_transforms(load_test=load_test)

        dataset_reader = datasets.CIFAR10 if self.config.variant == enums.CIFAR.CIFAR10 else datasets.CIFAR100
        if load_test:
            test_dataset = dataset_reader(root="./data", train=False, download=False, transform=self.test_transforms)
            self.test_dataloader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False)

            print("Test dataloader initialized successfully.")
            return

        # Apply transforms and initialize dataloaders for training and validation datasets.
        train_dataset = dataset_reader(root="./data", train=True, download=False, transform=self.train_transforms)
        val_dataset = dataset_reader(root="./data", train=True, download=False, transform=self.val_transforms)

        # Under xmp.spawn (multi-core TPU), the world size > 1.
        # Each core must see a disjoint shard: use DistributedSampler over a Subset.
        # Single-core TPU and CPU/GPU paths keep the original SubsetRandomSampler.
        if self._is_tpu and self._xla_world_size() > 1:
            train_subset = Subset(train_dataset, self._train_indices.indices)
            val_subset = Subset(val_dataset, self._val_indices.indices)
            train_sampler = DistributedSampler(
                train_subset,
                num_replicas=self._xla_world_size(),
                rank=self._xla_ordinal(),
                shuffle=True,
            )
            val_sampler = DistributedSampler(
                val_subset,
                num_replicas=self._xla_world_size(),
                rank=self._xla_ordinal(),
                shuffle=False,
            )
            self.train_dataloader = DataLoader(
                train_subset,
                batch_size=self.config.batch_size,
                shuffle=False,  # DistributedSampler handles shuffling.
                sampler=train_sampler,
            )
            self.val_dataloader = DataLoader(
                val_subset,
                batch_size=self.config.batch_size,
                shuffle=False,
                sampler=val_sampler,
            )
            self.train_dataloader = pl.MpDeviceLoader(self.train_dataloader, self.device)
            self.val_dataloader = pl.MpDeviceLoader(self.val_dataloader, self.device)

            print("Training (validation) dataloaders initialized successfully.")

            return

        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            sampler=SubsetRandomSampler(self._train_indices.indices),
        )
        self.val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            sampler=SubsetRandomSampler(self._val_indices.indices),
        )

        print("Training (validation) dataloaders initialized successfully.")

    def fit(self):
        """
        Fit the ResNet model on the training dataset and evaluate on the validation dataset if configured.
        """

        # Initialize the model if not already initialized.
        if self.model is None:
            self._init_model()

        self.model.initialize_weights()
        self.model.train()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.config.learning_rate[-1],
            momentum=0.9,
            weight_decay=1e-4,
            nesterov=self.config.optimizer == enums.Optimizer.NAGSGD,
        )

        # The paper specifies a learning rate schedule where
        # the learning rate is decayedby a factor of 10 at 50% and 75% of the total iterations.
        milestones = [int(self.config.iterations * 0.5), int(self.config.iterations * 0.75)]
        enable_warmup = len(self.config.learning_rate) > 1 > self.config.warmup_threshold
        print(f"Warmup enabled: {enable_warmup}")

        lr_lambda_func = partial(
            self.lr_lambda,
            learning_rate=self.config.learning_rate,
            milestones=milestones,
            enable_warmup=enable_warmup,
        )
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda_func)

        print(f"Starting training on {self.device}...")
        num_epochs = self.config.iterations // len(self.train_dataloader) + 1
        print(f"Total batches: {len(self.train_dataloader)}, Epochs: {num_epochs}\n")

        for epoch in range(num_epochs):
            running_loss = 0.0
            correct = 0
            total = 0

            pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for batch_idx, (inputs, targets) in enumerate(pbar):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss: torch.Tensor = criterion(outputs, targets)
                loss.backward()
                if self._is_tpu:
                    xm.optimizer_step(optimizer)  # Syncs gradients across TPU cores, then steps.
                    torch_xla.sync()  # Flush the lazy XLA graph so subsequent .item() calls don't stall.
                else:
                    optimizer.step()

                # Track metrics
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                scheduler.step()

                # Update progress bar with batch metrics.
                batch_acc = correct / total
                avg_loss = running_loss / (batch_idx + 1)

                pbar.set_postfix(
                    {
                        "accuracy": f"{batch_acc:.4f}",
                        "loss": f"{avg_loss:.4f}",
                        "lr": f"{scheduler.get_last_lr()[0]:.4f}",
                    }
                )

            # Evaluate on validation set at end of epoch.
            if self.config.train_val_split != 1:
                val_acc, val_loss = self.evaluate(self.val_dataloader, criterion)
                print(
                    f"Epoch {epoch+1} - accuracy: {batch_acc:.4f} - loss: {avg_loss:.4f}",
                    f"- val_accuracy: {val_acc:.4f} - val_loss: {val_loss:.4f}",
                )
            else:
                print(f"Epoch {epoch+1} - accuracy: {batch_acc:.4f} - loss: {avg_loss:.4f}")

        print("\nTraining completed!")

    def load(self, model_path: str):
        """
        Load a trained model from disk.

        Reads the ModelConfig JSON and weights .pth that were written by `save()`.
        The files are located by the same 6n+2 naming convention used during saving.
        """

        # Load ModelConfig from JSON.
        config_path = f"{model_path}.json"
        raw = json.loads(Path(config_path).read_text(encoding="utf-8"))
        model_config = ModelConfig(
            device=self.device,
            initial_out_planes=raw["initial_out_planes"],
            kernel_size=raw["kernel_size"],
            num_classes=raw["num_classes"],
            padding=raw["padding"],
            residual_block_depth=raw["residual_block_depth"],
            stride=raw["stride"],
        )

        # Reconstruct model and load weights.
        self.model: TorchResNet = TorchResNet(model_config).to(self.device)
        weights_path = f"{model_path}.pth"
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval()
        print(f"Model loaded from {weights_path}")

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
