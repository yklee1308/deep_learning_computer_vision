import torch
from torch import nn
from torch.nn import Module


class LeNet(Module):
    def __init__(self, in_channels, out_channels):
        super(LeNet, self).__init__()

        # 1st Layer : Conv1 (tanh) -> Average-pooling
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=6,
                               kernel_size=5, stride=1, padding=0)
        self.actv1 = nn.Tanh()
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        # 2nd Layer : Conv2 (tanh) -> Average-pooling
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16,
                               kernel_size=5, stride=1, padding=0)
        self.actv2 = nn.Tanh()
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        # 3rd Layer : Conv3 (tanh)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120,
                               kernel_size=5, stride=1, padding=0)
        self.actv3 = nn.Tanh()

        # Adaptive-average-pooling
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        # 4th Layer : FC4 (tanh)
        self.fc4 = nn.Linear(in_features=120, out_features=84)
        self.actv4 = nn.Tanh()

        # 5th Layer : FC5
        self.fc5 = nn.Linear(in_features=84, out_features=out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.actv1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.actv2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.actv3(x)

        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)

        x = self.fc4(x)
        x = self.actv4(x)

        x = self.fc5(x)

        return x
