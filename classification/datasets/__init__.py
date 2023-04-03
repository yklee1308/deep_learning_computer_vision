from .mnist import MNIST
from .imagenet import ImageNet

datasets = {'MNIST' : MNIST,
            'ImageNet' : ImageNet}

def getDataset(dataset):
    return datasets[dataset]
