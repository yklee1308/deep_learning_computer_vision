from matplotlib import pyplot as plt

import torch

from metric import getMetric


class Tester(object):
    def __init__(self, model, dataset, device, args):
        self.model = model
        self.dataset = dataset
        self.device = device

        self.batch_size = args.batch_size

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

            self.showInference(x, y, img_path=img_path, channels=self.dataset.img_shape[0])
            
    def loadModel(self, model, dataset):
        self.model.load_state_dict(torch.load('classification/weights/{}_{}_weights.pth'.format(model, dataset)))

    def showInference(self, x, y, img_path, channels):
        x, y, img_path = torch.argmax(x).item(), y[0].item(), img_path[0]

        img = plt.imread(fname=img_path)
        if channels == 1:
            plt.imshow(img, cmap='gray')
        elif channels == 3:
            plt.imshow(img)
            
        plt.title('[Prediction] {} [Label] {}'.format(self.dataset.classes[x], self.dataset.classes[y]))

        plt.show()
