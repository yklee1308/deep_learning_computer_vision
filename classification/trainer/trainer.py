import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD


class Trainer(object):
    def __init__(self, model, dataset, device, args):
        self.model = model
        self.dataset = dataset
        self.device = device

        # Loss Function
        if args.loss_function == 'cross_entropy':
            self.loss_function = CrossEntropyLoss().to(self.device)

        # Optimizer
        if args.optimizer == 'stochastic_gradient_descent':
            self.optimizer = SGD(params=self.model.parameters(), lr=args.learning_rate,
                                 momentum=args.momentum, weight_decay=args.weight_decay)
            
        self.epochs = args.epochs
        self.batch_size = args.batch_size

        if args.resume_training:
            self.loadModel(args.model, args.dataset)

    def train(self):
        self.model.train()

        for epoch in range(self.epochs):
            for i, (x, y) in enumerate(self.dataset.train_data):
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
            acc.append(correct_k.mul_(100 / self.batch_size))

        return acc
