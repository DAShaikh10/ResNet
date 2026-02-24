"""
@Author: DAShaikh10
"""

# pylint: disable=line-too-long, invalid-name

import argparse

from utils import enums
from resnet import ResNetConfig
from resnet.cifar.engines import TorchResNetEngine


# Read command-line arguments.
# NOTE: The default values and choices are set according to original ResNet paper (https://arxiv.org/pdf/1512.03385)
parser = argparse.ArgumentParser(description="Train Residual Network (ResNet) on CIFAR-10 or CIFAR-100 dataset.")
parser.add_argument(
    "--activation",
    "-act",
    type=enums.ActivationFunction,
    default=enums.ActivationFunction.RELU,
    help="Activation function to use in the model (Default: ReLU)",
    choices=[enum.value for enum in enums.ActivationFunction],
)
parser.add_argument("--batch_size", "-bs", type=int, default=128, help="Batch size for training (Default: 128)")
parser.add_argument(
    "--device",
    "-d",
    type=enums.Device,
    default=None,
    help="Device to use for training (Default: auto-detected)",
    choices=[None] + [enum.value for enum in enums.Device],
)
parser.add_argument(
    "--framework",
    "-f",
    type=enums.Framework,
    default=enums.Framework.PYTORCH,
    help="Deep learning framework to use (Default: PyTorch)",
    choices=[enum.value for enum in enums.Framework],
)
parser.add_argument(
    "--iterations", "-it", type=int, default=64_000, help="Number of training iterations (Default: 64,000)"
)
parser.add_argument(
    "--learning_rate",
    "-lr",
    type=float,
    default=[0.01, 0.1],
    nargs=2,
    help="Learning rate(s) for training. First learning rate will be used for warmup. (Default: [0.01, 0.1])",
)
parser.add_argument(
    "--optimizer",
    "-opt",
    type=enums.Optimizer,
    default=enums.Optimizer.SGD,
    help="Optimizer to use for training (Default: SGD)",
    choices=[enum.value for enum in enums.Optimizer],
)
# TODO: Maybe provide regularization configuration as well.
parser.add_argument(
    "--residual_block_depth",
    "-rd",
    type=int,
    default=18,
    help="Depth of the residual block (Default: 18)",
    choices=[3, 5, 7, 9, 18, 200],
)
parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed for reproducibility (Default: 42)")
parser.add_argument(
    "--train_val_split", "-tvs", type=float, default=0.9, help="Train-validation split ratio (Default: 0.9)"
)
parser.add_argument(
    "--variant",
    "-vt",
    type=enums.CIFAR,
    default=enums.CIFAR.CIFAR10,
    help="CIFAR dataset variant to use (Default: CIFAR-10)",
    choices=[enum.value for enum in enums.CIFAR],
)
parser.add_argument(
    "--weight_initialization",
    "-wi",
    type=enums.WeightInitialization,
    default=enums.WeightInitialization.HE,
    help="Weight initialization method (Default: He)",
    choices=[enum.value for enum in enums.WeightInitialization],
)
parser.add_argument(
    "--warmup_threshold",
    "-wt",
    type=float,
    default=0.8,
    help="The training accuracy threshold for warmup (Default: 0.8)",
)

# Parse the command-line arguments.
args = parser.parse_args()
activation: enums.ActivationFunction = args.activation
batch_size: int = args.batch_size
device: None | enums.Device = args.device
framework: enums.Framework = args.framework
iterations: int = args.iterations
learning_rate: list[float] = args.learning_rate
optimizer: enums.Optimizer = args.optimizer
residual_block_depth: int = args.residual_block_depth
seed: int = args.seed
train_val_split: float = args.train_val_split
variant: enums.CIFAR = args.variant
weight_initialization: enums.WeightInitialization = args.weight_initialization
warmup_threshold: float = args.warmup_threshold

# Quick sanity check for the parsed arguments.
assert 0 < train_val_split <= 1, f"Invalid train-validation split ratio: {train_val_split}"
assert 0 < warmup_threshold < 1, f"Invalid warmup threshold: {warmup_threshold}"

# Construct the ResNet configuration.
resnet_config = ResNetConfig(
    activation=activation,
    batch_size=batch_size,
    device=device,
    framework=framework,
    iterations=iterations,
    learning_rate=learning_rate,
    optimizer=optimizer,
    residual_block_depth=residual_block_depth,
    seed=seed,
    train_val_split=train_val_split,
    variant=variant,
    weight_initialization=weight_initialization,
    warmup_threshold=warmup_threshold,
)

# Instantiate the ResNet model with the specified configuration.
resnet: TorchResNetEngine = None
if resnet_config.framework == enums.Framework.PYTORCH:
    resnet = TorchResNetEngine(resnet_config)

# Initialize the dataloader for training (and validation) data. The test dataloader will be initialized after training is complete.
print(f"Initializing dataloaders for {resnet_config.variant.value} dataset.")
resnet.init_dataloader(load_test=False)
print("Dataloaders initialized successfully.")

# Start the training process.
resnet.fit()
