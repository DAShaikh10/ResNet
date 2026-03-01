"""
@Author: DAShaikh10
@Description: PyTorch TPU engine for training and evaluating Residual Networks (ResNets)
              on CIFAR-10 and CIFAR-100 datasets.
              Currentl setup does not support multi-core TPU. This will be added in the future.
"""

# pylint: disable=too-many-locals

import json

from pathlib import Path
from functools import partial

import torch
import torch.nn.functional as F

from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, TensorDataset, DistributedSampler
from torchvision import datasets

try:
    import torch_xla
    import torch_xla.runtime as xr
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl

    XLA_AVAILABLE = True
except ImportError:
    XLA_AVAILABLE = False

from utils import enums
from resnet import ResNetConfig

from resnet.cifar.models import ModelConfig, TorchResNet

from .gpu_engine import ResNetEngine


class _CachedCIFARDataset(Dataset):
    """
    Holds all images as pre-normalised `float32` tensors in RAM.

    Training augmentations (random-flip → pad → random-crop) run as pure
    tensor ops, eliminating the per-sample `numpy → PIL → PIL → Tensor`
    conversion chain that otherwise dominates CPU time on the TPU host.
    """

    def __init__(self, images: torch.Tensor, labels: torch.Tensor, augment: bool = False):
        self.images = images  # [N, 3, 32, 32] float32, already normalised.
        self.labels = labels  # [N] int64
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.augment:
            # Random horizontal flip (matches transforms.RandomHorizontalFlip()).
            if torch.rand(1).item() > 0.5:
                img = img.flip(-1)
            # Pad 4 px on each side → [3, 40, 40], then random 32×32 crop
            # (matches transforms.RandomCrop(32, padding=4)).
            img = F.pad(img, (4, 4, 4, 4))
            i = torch.randint(0, 9, (1,)).item()
            j = torch.randint(0, 9, (1,)).item()
            img = img[:, i : i + 32, j : j + 32]
        return img, self.labels[idx]


class TorchResNetTPUEngine(ResNetEngine):
    """
    PyTorch TPU engine for CIFAR ResNet.
    """

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
        self.model = torch.compile(self.model, backend="openxla")

        print(f"Model initialized and moved to {self.device}.")

    def _setup_device(self):
        """
        Setup the device for training. Auto-detect **GPU > CPU** when no device is specified.

        This must only be called **inside a spawned worker** (after
        `torch_xla.launch()`).  Calling it in the main process would
        initialise the XLA runtime and prevent multi-core spawning.
        """

        # Auto-detect device: only route to TPU if real hardware is confirmed.
        device = torch_xla.device()
        if xm.xla_device_hw(device) == "TPU":
            self.device = device
            world = xr.world_size()
            print(f"TPU detected: {self.device}")
            print(f"TPU cores (world size): {world}")
            print("Using TPU for training.")
            return

        self.device = torch.device("cpu")
        print("Warning: TPU requested but not available. Falling back to CPU.")

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
                outputs: torch.Tensor = self.model(inputs)
                total += targets.size(0)
                correct += outputs.argmax(1).eq(targets).sum()
                if criterion is not None:
                    total_loss += criterion(outputs, targets)

        # Single device → host sync for the whole evaluation pass.
        torch_xla.sync()
        correct = correct.item()
        if criterion is not None:
            total_loss = total_loss.item()

        accuracy = correct / total
        avg_loss = total_loss / len(dataloader) if criterion is not None else None

        self.model.train()  # Switch back to train mode.

        if criterion is not None:
            return accuracy, avg_loss
        return accuracy

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
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay,
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

        num_batches = len(self.train_dataloader)
        for epoch in range(num_epochs):
            running_loss = 0.0
            correct = 0
            total = 0

            for inputs, targets in self.train_dataloader:
                optimizer.zero_grad(set_to_none=True)

                outputs: torch.Tensor = self.model(inputs)
                loss: torch.Tensor = criterion(outputs, targets)
                loss.backward()

                xm.optimizer_step(optimizer)  # Syncs gradients across TPU cores, then steps.
                scheduler.step()

                # On TPU, accumulate metrics as device tensors to avoid a
                # host-sync round-trip (.item()) on every training step.
                running_loss += loss.detach()
                total += targets.size(0)
                correct += outputs.detach().argmax(1).eq(targets).sum()
                del outputs, loss  # Free computation graph immediately.

            # Materialise accumulated device tensors once per epoch on TPU.
            torch_xla.sync()
            running_loss = running_loss.item()
            correct = correct.item()

            batch_acc = correct / total
            avg_loss = running_loss / num_batches

            # Epoch-level logging (no per-batch tqdm to avoid Python overhead on TPU).
            lr = scheduler.get_last_lr()[0]
            if self.config.train_val_split != 1:
                val_acc, val_loss = self.evaluate(self.val_dataloader, criterion)
                print(
                    f"Epoch {epoch+1}/{num_epochs}"
                    f" - accuracy: {batch_acc:.4f} - loss: {avg_loss:.4f}"
                    f" - val_accuracy: {val_acc:.4f} - val_loss: {val_loss:.4f}"
                    f" - lr: {lr:.4f}",
                )
            else:
                print(
                    f"Epoch {epoch+1}/{num_epochs}"
                    f" - accuracy: {batch_acc:.4f} - loss: {avg_loss:.4f}"
                    f" - lr: {lr:.4f}",
                )

        print("\nTraining completed!")

    # NOTE: We don't use the last batch as it may not be of perfect shape and hence cause performance loss on TPU.
    # This is not proved from the research paper and hence you can turn it off for strict replication.
    def init_dataloader(self, load_test: bool = False, drop_last: bool = True):
        """
        Initialize the data loader for **CIFAR-10** or **CIFAR-100** dataset.

        All images are converted to normalised ``float32`` tensors **once** and cached
        in host RAM.  Training augmentations (flip, pad, crop) run as pure tensor ops,
        avoiding the per-sample ``numpy → PIL → Tensor`` overhead that otherwise
        starves the TPU of data.

        Args:
            drop_last: bool - Whether to drop the last incomplete batch. _(Default: True)_
                              This slightly improves performance on TPU device.
            load_test: bool - Whether to load the test dataset. _(Default: False)_
        """

        print("Downloading / Loading the dataset and computing statistics. This may take a while...")

        self._prepare_dataset(load_test=load_test)

        dataset_reader = datasets.CIFAR10 if self.config.variant == enums.CIFAR.CIFAR10 else datasets.CIFAR100

        if load_test:
            # Materialise the entire test set as a pre-normalised float32 tensor.
            raw = dataset_reader(root="./data", train=False, download=False)
            images = torch.from_numpy(raw.data).permute(0, 3, 1, 2).float() / 255.0
            labels = torch.tensor(raw.targets, dtype=torch.long)
            del raw  # Free the underlying numpy arrays.

            mean = self.test_mean.view(1, 3, 1, 1)
            std = self.test_std.view(1, 3, 1, 1)
            images = (images - mean) / std
            del mean, std  # Free intermediate normalisation tensors.

            test_dataset = TensorDataset(images, labels)
            del images, labels  # Backing data is now owned by the TensorDataset.

            if xr.world_size() > 1:
                test_sampler = DistributedSampler(
                    test_dataset,
                    num_replicas=xr.world_size(),
                    rank=xr.get_ordinal(),
                    seed=self.config.seed,
                    shuffle=False,
                )
                self.test_dataloader = DataLoader(
                    test_dataset,
                    batch_size=self.config.batch_size,
                    shuffle=False,
                    sampler=test_sampler,
                    drop_last=drop_last,
                    persistent_workers=True,
                )
            else:
                self.test_dataloader = DataLoader(
                    test_dataset,
                    batch_size=self.config.batch_size,
                    shuffle=False,
                    drop_last=drop_last,
                    persistent_workers=True,
                )
            self.test_dataloader = pl.MpDeviceLoader(self.test_dataloader, self.device)

            print("Test dataloader initialized successfully.")
            return

        # Materialise CIFAR into cached float32 tensors.
        raw = dataset_reader(root="./data", train=True, download=False)
        all_images = torch.from_numpy(raw.data).permute(0, 3, 1, 2).float() / 255.0
        all_labels = torch.tensor(raw.targets, dtype=torch.long)
        del raw  # Free the underlying numpy arrays.

        train_idx = self._train_indices.indices
        val_idx = self._val_indices.indices

        # Normalise each split with its own statistics (computed by _prepare_dataset).
        train_mean = self.train_mean.view(1, 3, 1, 1)
        train_std = self.train_std.view(1, 3, 1, 1)
        train_images = (all_images[train_idx] - train_mean) / train_std
        train_labels = all_labels[train_idx]

        val_mean = self.val_mean.view(1, 3, 1, 1)
        val_std = self.val_std.view(1, 3, 1, 1)
        val_images = (all_images[val_idx] - val_mean) / val_std
        val_labels = all_labels[val_idx]
        del all_images, all_labels  # Free the un-split copies.

        # Training dataset: cached tensors + fast tensor-op augmentations.
        train_dataset = _CachedCIFARDataset(train_images, train_labels, augment=True)
        # Validation dataset: cached tensors, no augmentation — zero per-sample work.
        val_dataset = TensorDataset(val_images, val_labels)
        del train_images, train_labels, val_images, val_labels  # Backing data now owned by datasets.

        # Under xmp.spawn (multi-core TPU), the world size > 1.
        # Each core must see a disjoint shard: use DistributedSampler.
        if xr.world_size() > 1:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=xr.world_size(),
                rank=xr.get_ordinal(),
                shuffle=True,
                seed=self.config.seed,
            )
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=xr.world_size(),
                rank=xr.get_ordinal(),
                seed=self.config.seed,
                shuffle=False,
            )
            self.train_dataloader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,  # DistributedSampler handles shuffling.
                sampler=train_sampler,
                drop_last=drop_last,
            )
            self.val_dataloader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                sampler=val_sampler,
                drop_last=drop_last,
            )

        # Single-core TPU and CPU path.
        else:
            self.train_dataloader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                generator=self._generator,
                drop_last=True,
            )
            self.val_dataloader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                drop_last=True,
            )

        # MpDeviceLoader pre-loads data to the XLA device, improving performance
        # for both single-core and multi-core TPU training.
        self.train_dataloader = pl.MpDeviceLoader(self.train_dataloader, self.device)
        self.val_dataloader = pl.MpDeviceLoader(self.val_dataloader, self.device)

        print("Training (validation) dataloaders initialized successfully.")

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

        # Reconstruct model on CPU, load weights, then move to XLA device once.
        self.model = TorchResNet(model_config)
        weights_path = f"{model_path}.pth"
        self.model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
        self.model.to(self.device)
        self.model = torch.compile(self.model, backend="openxla")
        self.model.eval()
        print(f"Model loaded from {weights_path}")
