"""
@Author: DAShaikh10
@Description: PyTorch engine for training and evaluating Residual Networks (ResNets)
              on GPU/CPU for CIFAR-10 and CIFAR-100 datasets.
"""

# pylint: disable=too-many-instance-attributes, too-many-locals

import json

from pathlib import Path
from functools import partial

import torch
import torchinfo

from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets


from utils import enums
from resnet import ResNetConfig

from resnet.cifar.models import ModelConfig, TorchResNet

from .engine import ResNetEngine


class TorchResNetEngine(ResNetEngine):
    """
    PyTorch engine for CIFAR ResNet.
    """

    def __init__(self, config: ResNetConfig):
        super().__init__(config)

        # Auto-detect and set device (GPU if available, else CPU).
        self._setup_device()

    def _setup_device(self):
        """
        Setup the device for training. Auto-detect **GPU > CPU** when no device is specified.
        """

        # Auto-detect device: only route to GPU if real hardware is confirmed.
        if self.config.device is None:
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
        print(f"{"No GPU detected. " if self.config.device is None else ""}Using CPU for training.")

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

        torchinfo.summary(self.model, input_size=(self.config.batch_size, 3, 32, 32), device=self.device)
        print("\n")

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
                total += targets.size(0)

                correct += outputs.argmax(1).eq(targets).sum().item()
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
        train_dataset = Subset(
            dataset_reader(root="./data", train=True, download=False, transform=self.train_transforms),
            self._train_indices.indices,
        )
        val_dataset = Subset(
            dataset_reader(root="./data", train=True, download=False, transform=self.val_transforms),
            self._val_indices.indices,
        )
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            generator=self._generator,
        )
        self.val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
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
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay,
            nesterov=self.config.optimizer == enums.Optimizer.NAGSGD,
        )

        # The paper specifies a learning rate schedule where
        # the learning rate is decayed by a factor of 10 at 50% and 75% of the total iterations.
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

                outputs: torch.Tensor = self.model(inputs)
                loss: torch.Tensor = criterion(outputs, targets)
                loss.backward()

                optimizer.step()
                scheduler.step()

                running_loss += loss.item()
                total += targets.size(0)
                correct += outputs.argmax(1).eq(targets).sum().item()
                del outputs, loss  # Free computation graph immediately.

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

        self.model = TorchResNet(model_config).to(self.device)
        weights_path = f"{model_path}.pth"
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device, weights_only=True))
        self.model.eval()
        print(f"Model loaded from {weights_path}")
