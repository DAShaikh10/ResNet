"""
@Author: DAShaikh10
"""

# pylint: disable=too-many-instance-attributes, too-many-positional-arguments, too-many-arguments

import torch
import torch.nn.functional as F

from torch import nn

from resnet.residual import BasicBlock
from .config import ModelConfig


class TorchResNet(nn.Module):
    """
    `PyTorch` implementation of Residual Network (ResNet) model for CIFAR-10 and CIFAR-100 datasets.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config

        # Initial convolutional layer.
        self.conv1 = nn.Conv2d(
            in_channels=3,  # CIFAR images have 3 color channels (RGB)
            out_channels=self.config.initial_out_planes,
            kernel_size=self.config.kernel_size,
            stride=1,  # The first layer does not downsample the input, so stride is set to 1.
            padding=self.config.padding,
            bias=False,  # Add REASON
        )
        self.bn1 = nn.BatchNorm2d(self.conv1.out_channels)

        # Stage 1.
        self.stage1 = self._build_residual_block(
            BasicBlock,
            depth=self.config.residual_block_depth,
            in_planes=self.config.initial_out_planes,
            out_planes=self.config.initial_out_planes,
            stride=1,
        )
        # Stage 2.
        self.stage2 = self._build_residual_block(
            BasicBlock,
            depth=self.config.residual_block_depth,
            in_planes=self.config.initial_out_planes,
            out_planes=self.config.initial_out_planes * 2,
            stride=self.config.stride,
        )
        # Stage 3.
        self.stage3 = self._build_residual_block(
            BasicBlock,
            depth=self.config.residual_block_depth,
            in_planes=self.config.initial_out_planes * 2,
            out_planes=self.config.initial_out_planes * 4,
            stride=self.config.stride,
        )

        # Global average pooling.
        # PyTorch adaptive average pooling serves as global average pooling layer
        # to ensure that the output feature maps are reduced to a fixed size of (1, 1),
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layer for classification.
        self.fc = nn.Linear(self.config.initial_out_planes * 4, self.config.num_classes)

    def _build_residual_block(self, block: BasicBlock, depth: int, in_planes: int, out_planes: int, stride: int):
        """
        Build a residual block with the specified input and output planes, and stride.
        """

        strides = [stride] + [1] * (depth - 1)
        layers = []
        current_in_planes = in_planes
        for s in strides:
            layers.append(block(current_in_planes, out_planes, self.config.kernel_size, s, self.config.padding))
            current_in_planes = out_planes  # Update for the next block in this stage
            self.in_planes = out_planes

        return nn.Sequential(*layers)

    def initialize_weights(self):
        """
        Initialize the weights of the model using the recommended initialization scheme.
        """

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                # Kaiming Normal (He Initialization)
                # The paper specifically recommends 'fan_out' for deep networks.
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                # Standard practice: scale (weight) to 1, shift (bias) to 0
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                # Section 3.4 implies standard initialization for the FC layer.
                # A normal distribution with std=0.01 is the standard 2015-era practice.
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)

        # A 2017 paper suggests better performance
        # by initializing the last batch normalization layer in each residual block to zero.
        # https://arxiv.org/abs/1706.02677
        for m in self.modules():
            if isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the ResNet model for CIFAR-10 or CIFAR-100.

        Args:
            x: Input tensor of shape (batch_size, 3, 32, 32)

        Returns:
            Logits tensor of shape (batch_size, num_classes) containing raw unnormalized
            class scores. These can be passed to CrossEntropyLoss (which applies softmax internally)
            or to softmax/argmax for inference.
        """

        # Initial convolutional layer, batch normalization, and ReLU activation.
        out: torch.Tensor = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        # Stages 1, 2, and 3 consisting of total `residual_stack_depth` number of `BasicBlocks`.
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)

        # Final global average pooling and fully connected layer for classification.
        out = self.global_avg_pool(out)

        return self.fc(out.flatten(1))  # Flatten the output for the fully connected layer.
