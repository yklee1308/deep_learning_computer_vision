from torch.nn import CrossEntropyLoss
from torch.nn import BCELoss
from torch.nn import MSELoss


def getCELoss(device):
    return CrossEntropyLoss().to(device)

def getBCELoss(device):
    return BCELoss().to(device)

def getMSELoss(device):
    return MSELoss().to(device)

loss_functions = {'CE' : getCELoss,
                  'BCE' : getBCELoss,
                  'MSE' : getMSELoss}

def getLossFunction(loss_function, device):
    multi_loss_function = list()
    for i in range(len(loss_function)):
        multi_loss_function.append(loss_functions[loss_function[i]](device=device))
    
    return multi_loss_function

def computeLoss(x, y, multi_loss_function, loss_weight):
    losses = list()
    for i in range(len(multi_loss_function)):
        loss = loss_weight[i] * multi_loss_function[i](x[i], y[i])
        losses.append(loss)
        
    loss = sum(losses)

    return loss
