from .mnist import Mnist
from .imagenet import ImageNet

datasets = {'MNIST' : Mnist,
            'ImageNet' : ImageNet}

def getDataset(dataset):
    return datasets[dataset]
