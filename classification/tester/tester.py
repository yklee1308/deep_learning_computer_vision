from matplotlib import pyplot as plt

import torch


class Tester(object):
    def __init__(self, model, dataset, device, args):
        self.model = model
        self.dataset = dataset
        self.device = device

        self.batch_size = args.batch_size

        self.loadModel(args.model, args.dataset)

    def test(self):
        self.model.eval()

        top_1s, top_5s = list(), list()
        with torch.no_grad():
            for i, (x, y, _) in enumerate(self.dataset.test_data):
                x, y = x.to(self.device), y.to(self.device)
                x = self.model(x)

                top_1 , top_5 = self.getAccuracy(x, y, top_k=(1, 5))
                top_1s.append(top_1)
                top_5s.append(top_5)

                print('[Batch] {}/{} [Top-1] {:.2f} [Top-5] {:.2f}' \
                      .format(i + 1, int(len(self.dataset.test_set) / self.batch_size) + 1, top_1, top_5))
                
        print('[Top-1] {:.2f} [Top-5] {:.2f}'.format(sum(top_1s) / len(top_1s), sum(top_5s) / len(top_5s)))

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
                
    def getAccuracy(self, x, y, top_k):
        _, x = x.topk(k=max(top_k), dim=1, largest=True, sorted=True)
        x = x.t()
        correct = x.eq(y.expand_as(other=x))

        acc = list()
        for k in top_k:
            correct_k = correct[:k].float().sum()
            acc.append(correct_k.mul_(100 / len(x.t())))

        return acc
