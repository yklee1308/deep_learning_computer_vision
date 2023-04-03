import torch
from torch import nn
from torch.nn import Module


class Inception(Module):
    def __init__(self, in_channels, out_channels):
        super(Inception, self).__init__()

        # 1st Branch : Conv1 (BN -> ReLU)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels[0],
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.norm1 = nn.BatchNorm2d(num_features=out_channels[0], eps=0.001)
        self.actv1 = nn.ReLU(inplace=True)

        # 2nd Branch : Conv2_1 (BN -> ReLU) -> Conv2_2 (BN -> ReLU)
        self.conv2_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels[1],
                                 kernel_size=1, stride=1, padding=0, bias=False)
        self.norm2_1 = nn.BatchNorm2d(num_features=out_channels[1], eps=0.001)
        self.actv2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(in_channels=out_channels[1], out_channels=out_channels[2],
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2_2 = nn.BatchNorm2d(num_features=out_channels[2], eps=0.001)
        self.actv2_2 = nn.ReLU(inplace=True)

        # 3rd Branch : Conv3_1 (BN -> ReLU) -> Conv3_2 (BN -> ReLU)
        self.conv3_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels[3],
                                 kernel_size=1, stride=1, padding=0, bias=False)
        self.norm3_1 = nn.BatchNorm2d(num_features=out_channels[3], eps=0.001)
        self.actv3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(in_channels=out_channels[3], out_channels=out_channels[4],
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.norm3_2 = nn.BatchNorm2d(num_features=out_channels[4], eps=0.001)
        self.actv3_2 = nn.ReLU(inplace=True)

        # 4th Branch : Max-pooling -> Conv4 (BN -> ReLU)
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True)
        self.conv4 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels[5],
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.norm4 = nn.BatchNorm2d(num_features=out_channels[5], eps=0.001)
        self.actv4 = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.norm1(x1)
        x1 = self.actv1(x1)

        x2 = self.conv2_1(x)
        x2 = self.norm2_1(x2)
        x2 = self.actv2_1(x2)
        x2 = self.conv2_2(x2)
        x2 = self.norm2_2(x2)
        x2 = self.actv2_2(x2)

        x3 = self.conv3_1(x)
        x3 = self.norm3_1(x3)
        x3 = self.actv3_1(x3)
        x3 = self.conv3_2(x3)
        x3 = self.norm3_2(x3)
        x3 = self.actv3_2(x3)

        x4 = self.pool4(x)
        x4 = self.conv4(x4)
        x4 = self.norm4(x4)
        x4 = self.actv4(x4)

        x = torch.cat([x1, x2, x3, x4], dim=1)

        return x

class GoogLeNet(Module):
    def __init__(self, in_channels, out_channels):
        super(GoogLeNet, self).__init__()

        # 1st Layer : Conv1 (BN -> ReLU) -> Max-pooling
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64,
                               kernel_size=7, stride=2, padding=3, bias=False)
        self.norm1 = nn.BatchNorm2d(num_features=64, eps=0.001)
        self.actv1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        # 2nd Stack : Conv2_1 (BN -> ReLU) -> Conv2_2 (BN -> ReLU) -> Max-pooling
        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=64,
                                 kernel_size=1, stride=1, padding=0, bias=False)
        self.norm2_1 = nn.BatchNorm2d(num_features=64, eps=0.001)
        self.actv2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(in_channels=64, out_channels=192,
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2_2 = nn.BatchNorm2d(num_features=192, eps=0.001)
        self.actv2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        # 3rd Stage : Inception3_1 -> Inception3_2 -> Max-pooling
        self.inception3_1 = Inception(in_channels=192, out_channels=(64, 96, 128, 16, 32, 32))
        self.inception3_2 = Inception(in_channels=256, out_channels=(128, 128, 192, 32, 96, 64))
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        # 4th Stage : Inception4_1 -> Inception4_2 -> Inception4_3 ->
        #             Inception4_4 -> Inception4_5 -> Max-pooling
        self.inception4_1 = Inception(in_channels=480, out_channels=(192, 96, 208, 16, 48, 64))
        self.inception4_2 = Inception(in_channels=512, out_channels=(160, 112, 224, 24, 64, 64))
        self.inception4_3 = Inception(in_channels=512, out_channels=(128, 128, 256, 24, 64, 64))
        self.inception4_4 = Inception(in_channels=512, out_channels=(112, 144, 288, 32, 64, 64))
        self.inception4_5 = Inception(in_channels=528, out_channels=(256, 160, 320, 32, 128, 128))
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        # 5th Stage : Inception5_1 -> Inception5_2
        self.inception5_1 = Inception(in_channels=832, out_channels=(256, 160, 320, 32, 128, 128))
        self.inception5_2 = Inception(in_channels=832, out_channels=(384, 192, 384, 48, 128, 128))

        # Adaptive_average_pooling
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        # 6th Layer : Dropout -> FC6
        self.drop6 = nn.Dropout(p=0.4)
        self.fc6 = nn.Linear(in_features=1024, out_features=out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.actv1(x)
        x = self.pool1(x)

        x = self.conv2_1(x)
        x = self.norm2_1(x)
        x = self.actv2_1(x)
        x = self.conv2_2(x)
        x = self.norm2_2(x)
        x = self.actv2_2(x)
        x = self.pool2(x)

        x = self.inception3_1(x)
        x = self.inception3_2(x)
        x = self.pool3(x)

        x = self.inception4_1(x)
        x = self.inception4_2(x)
        x = self.inception4_3(x)
        x = self.inception4_4(x)
        x = self.inception4_5(x)
        x = self.pool4(x)

        x = self.inception5_1(x)
        x = self.inception5_2(x)

        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)

        x = self.drop6(x)
        x = self.fc6(x)

        return x
