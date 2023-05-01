from torch.nn import CrossEntropyLoss


def getCELoss(device):
    return CrossEntropyLoss().to(device)

loss_functions = {'CE' : getCELoss}

def getLossFunction(loss_function):
    return loss_functions[loss_function]

def computeLoss(x, y, loss_function):
    return loss_function(x, y)
