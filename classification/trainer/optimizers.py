from torch.optim import SGD


def getSGD(args, model):
    return SGD(params=model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

optimizers = {'SGD' : getSGD}

def getOptimizer(optimizer, args, model):
    return optimizers[optimizer](args=args, model=model)
