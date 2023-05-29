import torch

from processings import getProcessing
from metrics import getMetric
from .loss_functions import getLossFunction
from .optimizers import getOptimizer
from .loss_functions import computeLoss


class Trainer(object):
    def __init__(self, model, dataset, device, args):
        self.model = model
        self.dataset = dataset
        self.device = device

        self.epochs = args.epochs
        self.batch_size = args.batch_size

        # Processing
        self.processing = getProcessing(args.model)(dataset=self.dataset)

        # Metric
        self.metric = getMetric(args.dataset)()

        # Loss Function
        self.loss_function = getLossFunction(args.loss_function, device=self.device)
        self.loss_weight = args.loss_weight

        # Optimizer
        self.optimizer = getOptimizer(args.optimizer, args=args, model=self.model)

        if args.resume_training:
            self.loadModel(args.model, args.dataset)

    def train(self):
        self.model.train()

        for epoch in range(self.epochs):
            for i, (x, y, _) in enumerate(self.dataset.train_data):
                x, y = self.processing.preprocess(x, y)

                x, y = x.to(self.device), y.to(self.device)
                x = self.model(x)

                loss = computeLoss(x, y, self.loss_function, self.loss_weight)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                x, y = self.processing.postprocess(x, y)

                acc = self.metric.computeAccuracy(x, y, mode='train')
                self.metric.printAccuracy(mode='train', epoch=(epoch + 1, self.epochs),
                                          batch=(i + 1, int(len(self.dataset.train_set) / self.batch_size) + 1), loss=loss, acc=acc)
                
    def loadModel(self, model, dataset):
        self.model.load_state_dict(torch.load('detection/weights/{}_{}_weights.pth'.format(model, dataset)))

    def saveModel(self, model, dataset):
        torch.save(self.model.state_dict(), 'detection/weights/{}_{}_weights.pth'.format(model, dataset))
