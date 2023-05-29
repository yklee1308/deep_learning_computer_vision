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

        self.num_regions = 128
        self.positive_ratio = 0.25
        self.iou_th = 0.5

    def preprocess(self, x, y):
        img, label = np.array(x[0], dtype=np.uint8), y[0]

        # Selective Search
        regions = self.runSelectiveSearch(img)

        x, y = list(), list(list() for i in range(2))
        
        # Positive Samples
        for region in regions:
            ious = list()
            for bbox in label[1]:
                iou = self.computeIoU(region, bbox)
                ious.append(iou)

            if max(ious) > self.iou_th:
                tl_x, tl_y, br_x, br_y = region
                sample = img[tl_y:br_y , tl_x:br_x]
                sample = transforms.ToPILImage()(sample)
                sample = transforms.Resize(size=self.img_shape[1:])(sample)
                sample = self.transform(sample)
                x.append(sample)

                label_idx = ious.index(max(ious))
                target_class, target_bbox = torch.tensor(label[0][label_idx]), torch.tensor(label[1][label_idx])
                y[0].append(target_class)
                y[1].append(target_bbox)

        num_positives = int(self.num_regions * self.positive_ratio)
        x = (x * (int(num_positives / len(x)) + 1))[:num_positives]
        for i in range(len(y)):
            y[i] = (y[i] * (int(num_positives / len(y[i])) + 1))[:num_positives] 

        # Negative Samples
        for region in regions:
            if region not in x and len(x) < self.num_regions:
                tl_x, tl_y, br_x, br_y = region
                sample = img[tl_y:br_y , tl_x:br_x]
                sample = transforms.ToPILImage()(sample)
                sample = transforms.Resize(size=self.img_shape[1:])(sample)
                sample = self.transform(sample)
                x.append(sample)

                target_class = torch.tensor(0)
                y[0].append(target_class)

        x, y = torch.stack(x, dim=0), list(torch.stack(target, dim=0) for target in y)

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

    def runSelectiveSearch(self, img):
        _, region_data = selective_search(img, scale=100, sigma=0.8, min_size=100)
        region_data = sorted(region_data, key=lambda x: x['size'], reverse=True)

        regions = list()
        for i in range(len(region_data)):
            x, y, w, h = region_data[i]['rect']
            region = (x, y, x + w + 1, y + h + 1)
            if region not in regions:
                regions.append(region)

        return regions
    
    def computeIoU(self, bbox1, bbox2):
        intersection = (max(bbox1[0], bbox2[0]), max(bbox1[1], bbox2[1]), min(bbox1[2], bbox2[2]), min(bbox1[3], bbox2[3]))

        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        intersection_area = max(intersection[2] - intersection[0], 0) * max(intersection[3] - intersection[1], 0)

        iou = intersection_area / (bbox1_area + bbox2_area - intersection_area)

        return iou
