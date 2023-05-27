import os
import numpy as np
from PIL import Image
from xml.etree.ElementTree import parse

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms


class VOC2007Custom(Dataset):
    def __init__(self, root, classes, loader, transform, target_transform):
        self.loader = loader
        self.transform = transform
        self.target_transform = target_transform

        self.samples, self.targets = self.makeDataset(path=root, classes=classes)

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        x, y = self.samples[idx], self.targets[idx]

        img_path = x
        x = self.loader(img_path=img_path)

        if self.transform != None:
            x = self.transform(x)

        if self.target_transform != None:
            y = self.target_transform(y)
            
        return x, y, img_path
    
    def makeDataset(self, path, classes):
        samples = os.listdir(path=path)
        for i in range(len(samples)):
            samples[i] = path + '/' + samples[i]

        targets = list(list() for i in range(len(samples)))
        for i in range(len(samples)):
            label_path = samples[i].replace('img', 'label').split('.')[0] + '.xml'
            tree = parse(source=label_path)
            root = tree.getroot()

            objects = root.findall('object')
            target = list(list() for i in range(2))
            for j in range(len(objects)):
                target_class = list(k for k, v in classes.items() if v == objects[j].find('name').text)[0]
                target_bbox = (int(objects[j].find('bndbox').find('xmin').text), int(objects[j].find('bndbox').find('ymin').text),
                               int(objects[j].find('bndbox').find('xmax').text), int(objects[j].find('bndbox').find('ymax').text))
                target[0].append(target_class)
                target[1].append(target_bbox)
                
            targets[i] = target

        return samples, targets

class VOC2007(object):
    def __init__(self, img_shape, batch_size, num_workers):
        self.dataset_path = 'C:/Datasets/VOC2007/'

        self.img_shape = img_shape
        self.num_classes = 21
        self.bbox_channels = 4

        self.classes = self.loadClasses(classes_path=self.dataset_path + 'VOC2007_classes.txt')

        self.loader = self.loadImage
        self.collator = self.collate

        self.norm = transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                         std=(0.229, 0.224, 0.225))
        
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             self.norm])

        # Train Set
        self.train_set = VOC2007Custom(root=self.dataset_path + 'VOC2007_img_train',
                                       loader=self.loader, classes=self.classes, transform=None, target_transform=None)

        self.train_data = DataLoader(dataset=self.train_set, batch_size=batch_size,
                                     shuffle=True, num_workers=num_workers, collate_fn=self.collator, pin_memory=True)

        # Test Set

        self.test_set = VOC2007Custom(root=self.dataset_path + 'VOC2007_img_test',
                                      loader=self.loader, classes=self.classes, transform=None, target_transform=None)

        self.test_data = DataLoader(dataset=self.test_set, batch_size=batch_size,
                                    shuffle=False, num_workers=num_workers, collate_fn=self.collator, pin_memory=True)
        
        # Inference Data
        self.inference_data = DataLoader(dataset=self.test_set, batch_size=1,
                                         shuffle=True, num_workers=num_workers, collate_fn=self.collator, pin_memory=True)

    def loadImage(self, img_path):
        with open(img_path, 'rb') as f:
            img = Image.open(f).convert(mode='RGB')

            return img
        
    def loadClasses(self, classes_path):
        classes = dict()
        with open(classes_path, 'r') as f:
            for i in range(self.num_classes):
                classes[i] = f.readline().rstrip()

        return classes
    
    def collate(self, batch):
        x, y, img_path = list(list() for i in range(3))
        for sample in batch:
            x.append(np.array(sample[0], dtype=np.uint8).tolist())
            y.append(sample[1])
            img_path.append(sample[2])

        return x, y, img_path
