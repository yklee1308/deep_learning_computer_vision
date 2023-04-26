from .rcnn import RCNN


models = {'R-CNN' : RCNN}

def getModel(model):
    return models[model]
