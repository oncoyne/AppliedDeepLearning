import torch

from utils import *
from torch import nn
from typing import NamedTuple
from torch.nn import functional as F

class ImageShape(NamedTuple):
    height: int
    width: int
    channels: int

class CNN(nn.Module):
    def __init__(self, height: int, width: int, channels: int):
        super().__init__()
        self.input = ImageShape(height=height, width=width, channels=channels)

        self.conv1 = nn.Conv2d(
            in_channels=self.input.channels,
            out_channels=32,
            kernel_size=(5, 5),
            padding=(2, 2),
        )
        self.initialise_layer(self.conv1)

        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv2 = nn.Conv2d(
            in_channels=self.conv1.out_channels,
            out_channels= 64,
            kernel_size=(3, 3),
            padding=(1, 1),
        )
        self.initialise_layer(self.conv2)

        self.pool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))

        self.conv3 = nn.Conv2d(
            in_channels=self.conv2.out_channels,
            out_channels= 128,
            kernel_size=(3, 3),
            padding=(1, 1),
        )
        self.initialise_layer(self.conv3)

        self.pool3 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))

        self.fc1 = nn.Linear(15488, 4608)
        self.initialise_layer(self.fc1)

        self.fc2 = nn.Linear(2304, 2304)
        self.initialise_layer(self.fc2)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.conv1(images)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool3(x)

        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.maxout(x)
        x = self.fc2(x)
        return x

    @staticmethod
    def maxout(x: torch.Tensor, axis: int=1, pool_size: int=2, pooling_function=torch.max) -> torch.Tensor:
        input_shape = tuple(x.size())
        num_feature_maps = input_shape[axis]
        num_feature_maps_out = num_feature_maps // pool_size
        pool_shape = (input_shape[:axis] + (num_feature_maps_out, pool_size) + input_shape[axis + 1:])
        input_reshaped = x.reshape(pool_shape)
        return pooling_function(input_reshaped, axis=axis + 1).values

    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "bias"):
            nn.init.constant_(layer.bias, 0.1)
        if hasattr(layer, "weight"):
            nn.init.normal_(layer.weight, mean=0.0, std=0.01)



