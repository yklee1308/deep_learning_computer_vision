from .rcnn_processing import RCNNProcessing


processings = {'R-CNN' : RCNNProcessing}

def getProcessing(model):
    return processings[model]
