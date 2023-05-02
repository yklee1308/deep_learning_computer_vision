import torch

from processing import getProcessing
from metric import getMetric


class Tester(object):
    def __init__(self, model, dataset, device, args):
        self.model = model
        self.dataset = dataset
        self.device = device

        self.batch_size = args.batch_size

        # Processing
        self.processing = getProcessing()(dataset=self.dataset)

        # Metric
        self.metric = getMetric()()

        self.loadModel(args.model, args.dataset)

    def test(self):
        self.model.eval()

        with torch.no_grad():
            for i, (x, y, _) in enumerate(self.dataset.test_data):
                x, y = x.to(self.device), y.to(self.device)
                x = self.model(x)

                acc = self.metric.computeAccuracy(x, y, mode='test')
                self.metric.printAccuracy(mode='test', batch=[i + 1, int(len(self.dataset.test_set) / self.batch_size) + 1], acc=acc)

        self.metric.printAccuracy(mode='end')

    def inference(self):
        self.model.eval()
        
        with torch.no_grad():
            x, y, img_path = next(iter(self.dataset.inference_data))
            x, y = x.to(self.device), y.to(self.device)
            x = self.model(x)

            self.processing.visualize(x, y, img_path=img_path)
            
    def loadModel(self, model, dataset):
        self.model.load_state_dict(torch.load('classification/weights/{}_{}_weights.pth'.format(model, dataset)))
