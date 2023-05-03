from .voc2007_metric import VOC2007Metric


metrics = {'VOC2007' : VOC2007Metric}

def getMetric(dataset):
    return metrics[dataset]
