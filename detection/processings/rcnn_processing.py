import numpy as np
from matplotlib import pyplot as plt
from selectivesearch import selective_search

import torch
from torch.nn.functional import one_hot
from torchvision import transforms
from torchvision.ops import nms


class RCNNProcessing(object):
    def __init__(self, dataset, args):
        self.input_shape = dataset.input_shape
        self.num_classes = dataset.num_classes
        self.classes = dataset.classes
        self.transform = dataset.transform

        self.num_regions = args.num_regions
        self.positive_ratio = args.positive_ratio
        self.iou_th = args.iou_th
        self.conf_score_th = args.conf_score_th

        self.region_proposals = None
        self.img_shape = None

    def preprocess(self, x, y=None):
        img, label = np.array(x[0], dtype=np.uint8), y[0]

        self.img_shape = img.shape

        # Selective Search
        regions = self.runSelectiveSearch(img)
        
        self.region_proposals = list()

        if y != None:
            x, y = list(), list(list() for i in range(2))
            
            # Positive Samples
            for region in regions:
                ious = list()
                for bbox in label[1]:
                    iou = self.computeIoU(region, bbox)
                    ious.append(iou)

                if max(ious) > self.iou_th:
                    sample = self.transformSample(img, transform=self.transform, region=region, input_shape=self.input_shape)
                    x.append(sample)

                    target_class = self.transformTargetClass(label, idx=ious.index(max(ious)), num_classes=self.num_classes)
                    target_bbox = self.transformTargetBbox(label, idx=ious.index(max(ious)), region=region)
                    y[0].append(target_class)
                    y[1].append(target_bbox)

                    self.region_proposals.append(region)

            if len(x) > 0:
                num_positives = int(self.num_regions * self.positive_ratio)
                for i in range(num_positives - len(x)):
                    x.append(x[int(i % num_positives)])
                    for j in range(len(y)):
                        y[j].append(y[j][int(i % num_positives)])
                    self.region_proposals.append(self.region_proposals[int(i % num_positives)])

            # Negative Samples
            for region in regions:
                if region not in x and len(x) < self.num_regions:
                    sample = self.transformSample(img, transform=self.transform, region=region, input_shape=self.input_shape)
                    x.append(sample)

                    target_class = self.transformTargetClass(label, idx=None, num_classes=self.num_classes)
                    y[0].append(target_class)

                    self.region_proposals.append(region)

            x, y = torch.stack(x, dim=0), list(torch.stack(target, dim=0) for target in y)

            return x, y
            
        else:
            x = list()
            for region in regions:
                if len(x) < self.num_regions:
                    sample = self.transformSample(img, transform=self.transform, region=region, input_shape=self.input_shape)
                    x.append(sample)

                    self.region_proposals.append(region)

            x = torch.stack(x, dim=0)

            return x
        
    def postprocess(self, x):
        class_bboxes, class_conf_scores = list(list() for i in range(self.num_classes)), list(list() for i in range(self.num_classes))
        for i, region in enumerate(self.region_proposals):
            for j in range(self.num_classes):
                conf_score = x[0][i][j]
                if conf_score > self.conf_score_th and j != 0:
                    bbox = self.inverseTransformBbox(x[1][i], region=region)
                    class_bboxes[j].append(bbox)
                    class_conf_scores[j].append(conf_score)

        x = list(list() for i in range(self.num_classes))
        for i in range(self.num_classes):
            if len(class_bboxes[i]) > 0:
                class_bboxes[i], class_conf_scores[i] = torch.stack(class_bboxes[i], dim=0), torch.stack(class_conf_scores[i], dim=0)

                # NMS
                bboxes = self.runNMS(class_bboxes[i], class_conf_scores[i], iou_th=self.iou_th)
                x[i] = bboxes

        return x

    def visualize(self, x, y, img_path):
        x, y, img_path = torch.argmax(x).item(), y[0].item(), img_path[0]

        img = plt.imread(fname=img_path)
        if self.channels == 1:
            plt.imshow(img, cmap='gray')
        elif self.channels == 3:
            plt.imshow(img)
            
        plt.title('[Prediction] {} [Label] {}'.format(self.classes[x], self.classes[y]))

        plt.show()

    def transformSample(self, img, transform, region, input_shape):
        tl_x, tl_y, br_x, br_y = region
        sample = img[tl_y:br_y, tl_x:br_x]
        sample = transforms.ToPILImage()(sample)
        sample = transforms.Resize(size=input_shape[1:])(sample)
        sample = transform(sample)

        return sample
    
    def transformTargetClass(self, label, idx, num_classes):
        if idx != None:
            target_class = one_hot(torch.tensor(label[0][idx]), num_classes=num_classes).float()
        else:
            target_class = one_hot(torch.tensor(0), num_classes=num_classes).float()

        return target_class
    
    def transformTargetBbox(self, label, idx, region):
        tl_x, tl_y, br_x, br_y = label[1][idx]
        bbox_w = br_x - tl_x
        bbox_h = br_y - tl_y
        bbox_x = tl_x + (bbox_w / 2)
        bbox_y = tl_y + (bbox_h / 2)

        tl_x, tl_y, br_x, br_y = region
        region_w = br_x - tl_x
        region_h = br_y - tl_y
        region_x = tl_x + (region_w / 2)
        region_y = tl_y + (region_h / 2)

        x = (bbox_x - region_x) / region_w
        y = (bbox_y - region_y) / region_h
        w = np.log(bbox_w / region_w)
        h = np.log(bbox_h / region_h)

        target_bbox = torch.tensor((x, y, w, h)).float()

        return target_bbox
    
    def inverseTransformBbox(self, bbox, region):
        bbox_x, bbox_y, bbox_w, bbox_h = bbox.tolist()

        tl_x, tl_y, br_x, br_y = region
        region_w = br_x - tl_x
        region_h = br_y - tl_y
        region_x = tl_x + (region_w / 2)
        region_y = tl_y + (region_h / 2)

        x = (bbox_x * region_w) + region_x
        y = (bbox_y * region_h) + region_y
        w = np.exp(bbox_w) * region_w
        h = np.exp(bbox_h) * region_h

        tl_x = int(x - (w / 2))
        tl_y = int(y - (h / 2))
        br_x = int(x + (w / 2))
        br_y = int(y + (h / 2))

        bbox = (tl_x, tl_y, br_x, br_y)

        return bbox

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
    
    def runNMS(self, bboxes, conf_scores, iou_th):
        bboxes = nms(bboxes, conf_scores, iou_threshold=iou_th)

        return bboxes
    
    def computeIoU(self, bbox1, bbox2):
        intersection = (max(bbox1[0], bbox2[0]), max(bbox1[1], bbox2[1]), min(bbox1[2], bbox2[2]), min(bbox1[3], bbox2[3]))

        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        intersection_area = max(intersection[2] - intersection[0], 0) * max(intersection[3] - intersection[1], 0)

        iou = intersection_area / (bbox1_area + bbox2_area - intersection_area)

        return iou
