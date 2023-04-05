import torch


class Tester(object):
    def __init__(self, model, dataset, device, args):
        self.model = model
        self.dataset = dataset
        self.device = device

        self.loadModel(args.model, args.dataset)

    def test(self):
        self.model.eval()

        top_1s, top_5s = list(), list()
        with torch.no_grad():
            for i, (x, y) in enumerate(self.dataset.train_data):
                x, y = x.to(self.device), y.to(self.device)
                x = self.model(x)

                top_1 , top_5 = self.getAccuracy(x, y, top_k=(1, 5))
                top_1s.append(top_1)
                top_5s.append(top_5)

                print('[Batch] {}/{} [Top-1] {:.2f} [Top-5] {:.2f}' \
                      .format(i + 1, int(len(self.dataset.train_set) / len(x)) + 1, top_1, top_5))
                
        print('[Top-1] {:.2f} [Top-5] {:.2f}'.format(sum(top_1s) / len(top_1s), sum(top_5s) / len(top_5s)))
                
    def loadModel(self, model, dataset):
        self.model.load_state_dict(torch.load('classification/weights/{}_{}_weights.pth'.format(model, dataset)))
                
    def getAccuracy(self, x, y, top_k):
        _, x = x.topk(k=max(top_k), dim=1, largest=True, sorted=True)
        x = x.t()
        correct = x.eq(y.expand_as(other=x))

        acc = list()
        for k in top_k:
            correct_k = correct[:k].float().sum()
            acc.append(correct_k.mul_(100 / len(x.t())))

        return acc
