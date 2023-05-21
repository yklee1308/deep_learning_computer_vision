import numpy as np
from matplotlib import pyplot as plt
from selectivesearch import selective_search

import torch


class RCNNProcessing(object):
    def __init__(self, dataset):
        self.classes = dataset.classes
        self.channels = dataset.img_shape[0]

        # Selective Search
        self.num_regions = 128

    def preprocess(self, x, y):
        regions = self.runSelectiveSearch(x[0], num_regions=self.num_regions)

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
        _, region_data = selective_search(np.array(img, dtype=np.uint8), scale=100, sigma=0.9, min_size=100)
        region_data = sorted(region_data, key=lambda x: x['size'], reverse=True)

        regions = list()
        for i in range(len(region_data)):
            region = [region_data[i]['rect'][0], region_data[i]['rect'][1],
                      region_data[i]['rect'][0] + region_data[i]['rect'][2],  region_data[i]['rect'][1] + region_data[i]['rect'][3]]
            if region not in regions:
                regions.append(region)

        regions = regions[:num_regions]

        return regions
