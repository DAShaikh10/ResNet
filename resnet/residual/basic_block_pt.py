"""
@Author: DAShaikh10
@Description: Basic residual block implementation for Residual Network (ResNet) in PyTorch,
              using Option A (identity shortcut with zero-padding for dimension mismatch)
              as described in the original research paper.
              This block is more suitable for CIFAR datasets, as it is computationally efficient and
              does not introduce additional parameters.
"""

# pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-instance-attributes

import torch
import torch.nn.functional as F

from torch import nn


class BasicBlock(nn.Module):
    """
    Basic residual block for Residual Network _(ResNet)_.

    The research paper authors proposed two options for the identity shortcut connection
    when the dimensions of the input and output do not match:
    - `Option A`: Identity shortcut with zero-padding for dimension mismatch _(used in CIFAR-10)_.
    - `Option B`: Projection shortcut using 1x1 convolution to match dimensions _(used in ImageNet)_.

    This is the basic block implementation which uses `Option A`, more suitable for **CIFAR** datasets,
    as it is computationally efficient and does not introduce additional parameters.
    """

    def __init__(self, in_planes: int, out_planes: int, kernel_size: int, stride: int, padding: int):
        super().__init__()

        # Use the prescribed stride for the first convolutional layer in the block.
        # This layer is responsible for downsampling the feature maps when needed (e.g., when stride > 1).
        self.conv1 = nn.Conv2d(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(self.conv1.out_channels)

        # The second convolutional layer in the block always has a stride of 1, as it does not perform downsampling.
        self.conv2 = nn.Conv2d(
            in_channels=out_planes,
            out_channels=out_planes,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(self.conv2.out_channels)

        # Option A: Identity shortcut with zero-padding for dimension mismatch.
        # Using Option A as per the paper for CIFAR: zero-padding for increasing dimensions.
        self.stride = stride
        self.in_planes = in_planes
        self.out_planes = out_planes

        # Create shortcut module.
        if stride != 1 or in_planes != out_planes:
            # Need to adjust the shortcut to match dimensions.
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor):
        """
        Forward pass through the basic residual block.
        """

        # First conv. layer with downsampling, batch normalization and ReLU activation.
        out: torch.Tensor = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        # Second conv. layer with batch normalization.
        out = self.conv2(out)
        out = self.bn2(out)

        # Finally, add the output of the second conv. layer to the shortcut connection (identity mapping).
        # F(x) = H(x) - x => H(x) = F(x) + x
        # Apply downsampling if stride != 1
        identity = x
        if self.stride != 1 or self.in_planes != self.out_planes:
            # Downsample spatial dimensions if needed.
            if self.stride != 1:
                identity = identity[:, :, :: self.stride, :: self.stride]
            # Pad channels if needed (Option A: zero-padding)
            if self.in_planes != self.out_planes:
                identity = F.pad(identity, (0, 0, 0, 0, 0, self.out_planes - self.in_planes))

        out += identity

        # As mentioned in the research paper,
        # the ReLU activation is applied after the addition of the shortcut connection.
        return F.relu(out)
