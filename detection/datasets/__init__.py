from .voc2007 import VOC2007


datasets = {'VOC2007' : VOC2007}

def getDataset(dataset):
    return datasets[dataset]
