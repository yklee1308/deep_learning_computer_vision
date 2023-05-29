from matplotlib import pyplot as plt

import torch


class Processing(object):
    def __init__(self, dataset):
        self.img_shape = dataset.img_shape
        self.classes = dataset.classes

    def visualize(self, x, y, img_path):
        x, y, img_path = torch.argmax(x).item(), y[0].item(), img_path[0]

        img = plt.imread(fname=img_path)
        if self.img_shape[0] == 1:
            plt.imshow(img, cmap='gray')
        elif self.img_shape[0] == 3:
            plt.imshow(img)
            
        plt.title('[Prediction] {} [Label] {}'.format(self.classes[x], self.classes[y]))

        plt.show()
