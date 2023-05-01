from torch.nn import CrossEntropyLoss


def getCELoss():
    return CrossEntropyLoss()

loss_functions = {'CE' : getCELoss}

def getLossFunction(loss_function):
    return loss_functions[loss_function]
