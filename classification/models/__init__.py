from .lenet import LeNet
from .alexnet import AlexNet
from .vggnet import VGGNet
from .googlenet import GoogLeNet
from .resnet import ResNet


models = {'LeNet' : LeNet,
          'AlexNet' : AlexNet,
          'VGGNet' : VGGNet,
          'GoogLeNet' : GoogLeNet,
          'ResNet' : ResNet}

def getModel(model):
    return models[model]
