import torch

from trainer.loss_functions import getLossFunction
from trainer.optimizers import getOptimizer


class Trainer(object):
    def __init__(self, model, dataset, device, args):
        self.model = model
        self.dataset = dataset
        self.device = device

        self.epochs = args.epochs
        self.batch_size = args.batch_size

        # Loss Function
        self.loss_function = getLossFunction(args.loss_function)().to(self.device)

        # Optimizer
        self.optimizer = getOptimizer(args.optimizer)(args, self.model)

        if args.resume_training:
            self.loadModel(args.model, args.dataset)

    def train(self):
        self.model.train()

        for epoch in range(self.epochs):
            for i, (x, y, _) in enumerate(self.dataset.train_data):
                x, y = x.to(self.device), y.to(self.device)
                x = self.model(x)

                loss = self.loss_function(x, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                top_1 , top_5 = self.getAccuracy(x, y, top_k=(1, 5))

                print('[Epoch] {}/{} [Batch] {}/{} [Loss] : {:.4f} [Top-1] {:.2f} [Top-5] {:.2f}' \
                      .format(epoch, self.epochs, i + 1, int(len(self.dataset.train_set) / self.batch_size) + 1, \
                              loss, top_1, top_5))
                
    def loadModel(self, model, dataset):
        self.model.load_state_dict(torch.load('classification/weights/{}_{}_weights.pth'.format(model, dataset)))

    def saveModel(self, model, dataset):
        torch.save(self.model.state_dict(), 'classification/weights/{}_{}_weights.pth'.format(model, dataset))
                
    def getAccuracy(self, x, y, top_k):
        _, x = x.topk(k=max(top_k), dim=1, largest=True, sorted=True)
        x = x.t()
        correct = x.eq(y.expand_as(other=x))

        acc = list()
        for k in top_k:
            correct_k = correct[:k].float().sum()
            acc.append(correct_k.mul_(100 / len(x.t())))

        return acc
