import torch
from torch import nn
from torch.nn import Module


class VGGNet(Module):
    def __init__(self, in_channels, out_channels):
        super(VGGNet, self).__init__()

        # 1st Stack : Conv1_1 (ReLU) -> Conv1_2 (ReLU) -> Max-pooling
        self.conv1_1 = nn.Conv2d(in_channels=in_channels, out_channels=64,
                                 kernel_size=3, stride=1, padding=1)
        self.actv1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64,
                                 kernel_size=3, stride=1, padding=1)
        self.actv1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 2nd Stack : Conv2_1 (ReLU) -> Conv2_2 (ReLU) -> Max-pooling
        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128,
                                 kernel_size=3, stride=1, padding=1)
        self.actv2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128,
                                 kernel_size=3, stride=1, padding=1)
        self.actv2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 3rd Stack : Conv3_1 (ReLU) -> Conv3_2 (ReLU) -> Conv3_3 (ReLU) -> Max-pooling
        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256,
                                 kernel_size=3, stride=1, padding=1)
        self.actv3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256,
                                 kernel_size=3, stride=1, padding=1)
        self.actv3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=256,
                                 kernel_size=3, stride=1, padding=1)
        self.actv3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 4th Stack : Conv4_1 (ReLU) -> Conv4_2 (ReLU) -> Conv4_3 (ReLU) -> Max-pooling
        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512,
                                 kernel_size=3, stride=1, padding=1)
        self.actv4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512,
                                 kernel_size=3, stride=1, padding=1)
        self.actv4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(in_channels=512, out_channels=512,
                                 kernel_size=3, stride=1, padding=1)
        self.actv4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 5th Stack : Conv5_1 (ReLU) -> Conv5_2 (ReLU) -> Conv5_3 (ReLU) -> Max-pooling
        self.conv5_1 = nn.Conv2d(in_channels=512, out_channels=512,
                                 kernel_size=3, stride=1, padding=1)
        self.actv5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(in_channels=512, out_channels=512,
                                 kernel_size=3, stride=1, padding=1)
        self.actv5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(in_channels=512, out_channels=512,
                                 kernel_size=3, stride=1, padding=1)
        self.actv5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Adaptive-average-pooling
        self.pool = nn.AdaptiveAvgPool2d(output_size=(7, 7))

        # 6th Layer : FC6 (ReLU) -> Dropout
        self.fc6 = nn.Linear(in_features=7 * 7 * 512, out_features=4096)
        self.actv6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout(p=0.5)

        # 7th Layer : FC7 (ReLU) -> Dropout
        self.fc7 = nn.Linear(in_features=4096, out_features=4096)
        self.actv7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout(p=0.5)

        # 8th Layer : FC8
        self.fc8 = nn.Linear(in_features=4096, out_features=out_channels)

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.actv1_1(x)
        x = self.conv1_2(x)
        x = self.actv1_2(x)
        x = self.pool1(x)

        x = self.conv2_1(x)
        x = self.actv2_1(x)
        x = self.conv2_2(x)
        x = self.actv2_2(x)
        x = self.pool2(x)

        x = self.conv3_1(x)
        x = self.actv3_1(x)
        x = self.conv3_2(x)
        x = self.actv3_2(x)
        x = self.conv3_3(x)
        x = self.actv3_3(x)
        x = self.pool3(x)

        x = self.conv4_1(x)
        x = self.actv4_1(x)
        x = self.conv4_2(x)
        x = self.actv4_2(x)
        x = self.conv4_3(x)
        x = self.actv4_3(x)
        x = self.pool4(x)

        x = self.conv5_1(x)
        x = self.actv5_1(x)
        x = self.conv5_2(x)
        x = self.actv5_2(x)
        x = self.conv5_3(x)
        x = self.actv5_3(x)
        x = self.pool5(x)

        x = self.pool(x)
        
        x = torch.flatten(x, start_dim=1)

        x = self.fc6(x)
        x = self.actv6(x)
        x = self.drop6(x)

        x = self.fc7(x)
        x = self.actv7(x)
        x = self.drop7(x)

        x = self.fc8(x)

        return x
