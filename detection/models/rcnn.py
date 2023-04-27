import torch
from torch import nn
from torch.nn import Module


class AlexNetExtractor(Module):
    def __init__(self, in_channels):
        super(AlexNetExtractor, self).__init__()

        # 1st Layer : Conv1 (ReLU) -> Max-pooling
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64,
                               kernel_size=11, stride=4, padding=2)
        self.actv1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        # 2nd Layer : Conv2 (ReLU) -> Max-pooling
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=192,
                               kernel_size=5, stride=1, padding=2)
        self.actv2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        # 3rd Layer : Conv3 (ReLU)
        self.conv3 = nn.Conv2d(in_channels=192, out_channels=384,
                               kernel_size=3, stride=1, padding=1)
        self.actv3 = nn.ReLU(inplace=True)

        # 4th Layer : Conv4 (ReLU)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=256,
                               kernel_size=3, stride=1, padding=1)
        self.actv4 = nn.ReLU(inplace=True)

        # 5th Layer : Conv5 (ReLU) -> Max-pooling
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256,
                               kernel_size=3, stride=1, padding=1)
        self.actv5 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2)

        # Adaptive-average-pooling
        self.pool = nn.AdaptiveAvgPool2d(output_size=(6, 6))

    def forward(self, x):
        x = self.conv1(x)
        x = self.actv1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.actv2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.actv3(x)

        x = self.conv4(x)
        x = self.actv4(x)

        x = self.conv5(x)
        x = self.actv5(x)
        x = self.pool5(x)

        x = self.pool(x)

        return x

class Classifier(Module):
    def __init__(self, out_channels):
        super(Classifier, self).__init__()

        # 1st Layer : FC1 (ReLU) -> Dropout
        self.fc1 = nn.Linear(in_features=6 * 6 * 256, out_features=4096)
        self.actv1 = nn.ReLU(inplace=True)
        self.drop1 = nn.Dropout(p=0.5)

        # 2nd Layer : FC2
        self.fc2 = nn.Linear(in_features=4096, out_features=out_channels)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)

        x = self.fc1(x)
        x = self.actv1(x)
        x = self.drop1(x)

        x = self.fc2(x)

        return x

class BboxRegressor(Module):
    def __init__(self, out_channels):
        super(BboxRegressor, self).__init__()

        # Layer : FC1
        self.fc1 = nn.Linear(in_features=6 * 6 * 256, out_features=out_channels)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)

        x = self.fc1(x)

        return x

class RCNN(Module):
    def __init__(self, in_channels, out_channels):
        super(RCNN, self).__init__()

        self.alexnet_extractor = AlexNetExtractor(in_channels=in_channels)
        self.classifier = Classifier(out_channels=out_channels[0])
        self.bbox_regressor = BboxRegressor(out_channels=out_channels[1])
    
    def forward(self, x):
        x = self.alexnet_extractor(x)
        
        x_class = self.classifier(x)

        x_bbox = self.bbox_regressor(x)

        return x_class, x_bbox
