"""
@Author: DAShaikh10
"""

# pylint: disable=too-many-instance-attributes, too-many-locals

from functools import partial

import torch
import torchinfo

from tqdm import tqdm
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, random_split, SubsetRandomSampler
from torchvision import datasets, transforms

from utils import enums
from resnet.config import ResNetConfig
from resnet.cifar.models import ModelConfig, TorchResNet


class TorchResNetEngine:
    """
    PyTorch engine for CIFAR ResNet.
    """

    def __init__(self, config: ResNetConfig):
        self._generator = torch.random.manual_seed(config.seed)
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

    def _setup_device(self):
        """
        Setup the device for training. Auto-detects GPU if available, otherwise uses CPU.
        """

        if self.config.device is None:
            # Auto-detect device
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                print(f"GPU detected: {torch.cuda.get_device_name(0)}")
                print("Using GPU for training.")
            else:
                self.device = torch.device("cpu")
                print("No GPU detected. Using CPU for training.")
        else:
            # Convert enum to torch device.
            device_str = self.config.device.value.lower()
            if device_str == "gpu":
                if torch.cuda.is_available():
                    self.device = torch.device("cuda")
                    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
                    print("Using GPU for training.")
                else:
                    print("Warning: GPU requested but not available. Falling back to CPU.")
                    self.device = torch.device("cpu")
            else:
                self.device = torch.device("cpu")
                print("Using CPU for training.")

    def _compute_statistics(self, dataloader: DataLoader):
        """
        Compute the mean and standard deviation of the dataset for normalization.
        """

        mean = 0.0
        std = 0.0
        total_samples = 0
        for images, _ in dataloader:
            batch_samples = images.size(0)
            images = images.view(batch_samples, images.size(1), -1)
            mean += images.mean(2).sum(0)
            std += images.std(2).sum(0)
            total_samples += batch_samples

        mean /= total_samples
        std /= total_samples

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

        torchinfo.summary(self.model, input_size=(self.config.batch_size, 3, 32, 32), device=str(self.device))
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

        self.model.initialize_weights()
        self.model.train()

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
                    f"- val_accuracy: {val_acc:.4f} - val_loss: {val_loss:.4f}"
                )
            else:
                print(f"Epoch {epoch+1} - accuracy: {batch_acc:.4f} - loss: {avg_loss:.4f}")

        print("\nTraining completed!")
