"""
@Author: DAShaikh10
@Description: Script to test / evaluate a trained Residual Network (ResNet) on CIFAR-10 or CIFAR-100 dataset.
"""

# pylint: disable=line-too-long

import argparse

from utils import enums
from resnet import ResNetConfig
from resnet.cifar.engines import TorchResNetEngine


# Read command-line arguments.
parser = argparse.ArgumentParser(
    description="Evaluate a trained Residual Network (ResNet) on CIFAR-10 or CIFAR-100 test set."
)
parser.add_argument("--batch_size", "-bs", type=int, default=128, help="Batch size for evaluation (Default: 128)")
parser.add_argument(
    "--device",
    "-d",
    type=enums.Device,
    default=None,
    help="Device to use for evaluation (Default: auto-detected)",
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
parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed for reproducibility (Default: 42)")
parser.add_argument(
    "--model_path",
    "-md",
    type=str,
    help="Model path to the saved model weights and config",
)
parser.add_argument(
    "--variant",
    "-vt",
    type=enums.CIFAR,
    default=enums.CIFAR.CIFAR10,
    help="CIFAR dataset variant to use (Default: CIFAR-10)",
    choices=[enum.value for enum in enums.CIFAR],
)

# Parse the command-line arguments.
args = parser.parse_args()
batch_size: int = args.batch_size
device: None | enums.Device = args.device
framework: enums.Framework = args.framework
seed: int = args.seed
model_path: str = args.model_path
variant: enums.CIFAR = args.variant

# Construct the ResNet configuration. `residual_block_depth` and `variant` are
# populated automatically by `load()` from the saved model's JSON config.
resnet_config = ResNetConfig(
    batch_size=batch_size,
    device=device,
    framework=framework,
    residual_block_depth=0,
    seed=seed,
    variant=variant,
)

# Instantiate the engine and load the trained model from disk.
resnet: TorchResNetEngine = None
if resnet_config.framework == enums.Framework.PYTORCH:
    resnet = TorchResNetEngine(resnet_config)

resnet.load(model_path=model_path)

# Initialize the test dataloader.
print(f"Initializing test dataloader for {resnet_config.variant.value} dataset.")
resnet.init_dataloader(load_test=True)

# Evaluate on the test set.
# We present the same accuracy and error rate metrics as the original research paper for comparison.
test_acc = resnet.evaluate(resnet.test_dataloader)
print(
    f"\nTest accuracy: {test_acc:.4f} ({test_acc * 100:.2f}%), Error rate: {1 - test_acc:.4f} ({(1 - test_acc) * 100:.2f}%)"
)
