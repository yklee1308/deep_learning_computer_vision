import numpy as np
from matplotlib import pyplot as plt
from selectivesearch import selective_search

import torch
from torchvision import transforms


class RCNNProcessing(object):
    def __init__(self, dataset):
        self.img_shape = dataset.img_shape
        self.classes = dataset.classes
        self.transform = dataset.transform

        # Selective Search
        self.num_regions = 128

    def preprocess(self, x, y):
        img, label = np.array(x[0], dtype=np.uint8), y[0]

        regions = self.runSelectiveSearch(img, num_regions=self.num_regions)

        x, y = list(), list()
        for i in range(self.num_regions):
            tl_x, tl_y, br_x, br_y = regions[i]
            sample = img[tl_y:br_y + 1, tl_x:br_x + 1]
            sample = transforms.ToPILImage()(sample)
            sample = transforms.Resize(size=self.img_shape[1:])(sample)
            sample = self.transform(sample)

            x.append(sample)

        x = torch.stack(x, dim=0)

        return x, y

    def visualize(self, x, y, img_path):
        x, y, img_path = torch.argmax(x).item(), y[0].item(), img_path[0]

        img = plt.imread(fname=img_path)
        if self.channels == 1:
            plt.imshow(img, cmap='gray')
        elif self.channels == 3:
            plt.imshow(img)
            
        plt.title('[Prediction] {} [Label] {}'.format(self.classes[x], self.classes[y]))

        plt.show()

    def runSelectiveSearch(self, img, num_regions):
        _, region_data = selective_search(img, scale=100, sigma=0.9, min_size=100)
        region_data = sorted(region_data, key=lambda x: x['size'], reverse=True)

        regions = list()
        for i in range(len(region_data)):
            x, y, w, h = region_data[i]['rect']
            region = [x, y, x + w , y + h]
            if region not in regions:
                regions.append(region)

        regions = regions[:num_regions]

        return regions
