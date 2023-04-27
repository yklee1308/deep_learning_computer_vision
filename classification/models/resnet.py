import torch
from torch import nn
from torch.nn import Module


class Bottleneck(Module):
    def __init__(self, in_channels, out_channels, stride, increase_dim):
        super(Bottleneck, self).__init__()

        # 1st Branch : Conv1_1 (BN -> ReLU) -> Conv1_2 (BN -> ReLU) -> Conv1_3 (BN)
        self.conv1_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels[0],
                                 kernel_size=1, stride=1, padding=0, bias=False)
        self.norm1_1 = nn.BatchNorm2d(num_features=out_channels[0])
        self.actv1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(in_channels=out_channels[0], out_channels=out_channels[1],
                                 kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1_2 = nn.BatchNorm2d(num_features=out_channels[1])
        self.actv1_2 = nn.ReLU(inplace=True)
        self.conv1_3 = nn.Conv2d(in_channels=out_channels[1], out_channels=out_channels[2],
                                 kernel_size=1, stride=1, padding=0, bias=False)
        self.norm1_3 = nn.BatchNorm2d(num_features=out_channels[2])

        # 2nd Branch : Conv2 (BN)
        if increase_dim == True:
            self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels[2],
                                   kernel_size=1, stride=stride, padding=0, bias=False)
            self.norm2 = nn.BatchNorm2d(num_features=out_channels[2])

        # ReLU
        self.actv = nn.ReLU(inplace=True)

    def forward(self, x, increase_dim):
        x1 = self.conv1_1(x)
        x1 = self.norm1_1(x1)
        x1 = self.actv1_1(x1)
        x1 = self.conv1_2(x1)
        x1 = self.norm1_2(x1)
        x1 = self.actv1_2(x1)
        x1 = self.conv1_3(x1)
        x1 = self.norm1_3(x1)

        x2 = x
        if increase_dim == True:
            x2 = self.conv2(x2)
            x2 = self.norm2(x2)

        x = x1 + x2
        x = self.actv(x)

        return x

class ResNet(Module):
    def __init__(self, in_channels, out_channels):
        super(ResNet, self).__init__()

        # 1st Layer : Conv1 (BN -> ReLU) -> Max-pooling
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64,
                               kernel_size=7, stride=2, padding=3, bias=False)
        self.norm1 = nn.BatchNorm2d(num_features=64)
        self.actv1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 2nd Stage : Bottleneck2_1 -> Bottleneck2_2 -> Bottleneck2_3
        self.bottleneck2_1 = Bottleneck(in_channels=64, out_channels=(64, 64, 256),
                                        stride=1, increase_dim=True)
        self.bottleneck2_2 = Bottleneck(in_channels=256, out_channels=(64, 64, 256),
                                        stride=1, increase_dim=False)
        self.bottleneck2_3 = Bottleneck(in_channels=256, out_channels=(64, 64, 256),
                                        stride=1, increase_dim=False)

        # 3rd Stage : Bottleneck3_1 -> Bottleneck3_2 -> Bottleneck3_3 -> Bottleneck3_4
        self.bottleneck3_1 = Bottleneck(in_channels=256, out_channels=(128, 128, 512),
                                        stride=2, increase_dim=True)
        self.bottleneck3_2 = Bottleneck(in_channels=512, out_channels=(128, 128, 512),
                                        stride=1, increase_dim=False)
        self.bottleneck3_3 = Bottleneck(in_channels=512, out_channels=(128, 128, 512),
                                        stride=1, increase_dim=False)
        self.bottleneck3_4 = Bottleneck(in_channels=512, out_channels=(128, 128, 512),
                                        stride=1, increase_dim=False)

        # 4th Stage : Bottleneck4_1 -> Bottleneck4_2 -> Bottleneck4_3 ->
        #             Bottleneck4_4 -> Bottleneck4_5 -> Bottleneck4_6
        self.bottleneck4_1 = Bottleneck(in_channels=512, out_channels=(256, 256, 1024),
                                        stride=2, increase_dim=True)
        self.bottleneck4_2 = Bottleneck(in_channels=1024, out_channels=(256, 256, 1024),
                                        stride=1, increase_dim=False)
        self.bottleneck4_3 = Bottleneck(in_channels=1024, out_channels=(256, 256, 1024),
                                        stride=1, increase_dim=False)
        self.bottleneck4_4 = Bottleneck(in_channels=1024, out_channels=(256, 256, 1024),
                                        stride=1, increase_dim=False)
        self.bottleneck4_5 = Bottleneck(in_channels=1024, out_channels=(256, 256, 1024),
                                        stride=1, increase_dim=False)
        self.bottleneck4_6 = Bottleneck(in_channels=1024, out_channels=(256, 256, 1024),
                                        stride=1, increase_dim=False)

        # 5th Stage : Bottleneck5_1 -> Bottleneck5_2 -> Bottleneck5_3
        self.bottleneck5_1 = Bottleneck(in_channels=1024, out_channels=(512, 512, 2048),
                                        stride=2, increase_dim=True)
        self.bottleneck5_2 = Bottleneck(in_channels=2048, out_channels=(512, 512, 2048),
                                        stride=1, increase_dim=False)
        self.bottleneck5_3 = Bottleneck(in_channels=2048, out_channels=(512, 512, 2048),
                                        stride=1, increase_dim=False)

        # Adaptive_average_pooling
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        # 6th Layer : FC6
        self.fc6 = nn.Linear(in_features=2048, out_features=out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.actv1(x)
        x = self.pool1(x)

        x = self.bottleneck2_1(x, increase_dim=True)
        x = self.bottleneck2_2(x, increase_dim=False)
        x = self.bottleneck2_3(x, increase_dim=False)

        x = self.bottleneck3_1(x, increase_dim=True)
        x = self.bottleneck3_2(x, increase_dim=False)
        x = self.bottleneck3_3(x, increase_dim=False)
        x = self.bottleneck3_4(x, increase_dim=False)

        x = self.bottleneck4_1(x, increase_dim=True)
        x = self.bottleneck4_2(x, increase_dim=False)
        x = self.bottleneck4_3(x, increase_dim=False)
        x = self.bottleneck4_4(x, increase_dim=False)
        x = self.bottleneck4_5(x, increase_dim=False)
        x = self.bottleneck4_6(x, increase_dim=False)

        x = self.bottleneck5_1(x, increase_dim=True)
        x = self.bottleneck5_2(x, increase_dim=False)
        x = self.bottleneck5_3(x, increase_dim=False)

        x = self.pool(x)
        
        x = torch.flatten(x, start_dim=1)

        x = self.fc6(x)

        return x
