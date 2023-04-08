from PIL import Image

from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder
from torchvision import transforms

from datasets.classes import imagenet_classes


class ImageNetCustom(DatasetFolder):
    def __init__(self, root, loader, extensions, transform, target_transform):
        super(ImageNetCustom, self).__init__(root=root, loader=loader, extensions=extensions, transform=transform, target_transform=target_transform)
    
    def __getitem__(self, idx):
        x, y = self.samples[idx], int(self.targets[idx])

        img_path = x[0]
        img = self.loader(img_path=img_path)
        x = self.transform(img)

        if self.target_transform != None:
            y = self.target_transform(y)
            
        return x, y, img_path

class ImageNet(object):
    def __init__(self, img_shape, batch_size, num_workers):
        self.img_shape = img_shape
        self.num_classes = 1000

        self.classes = imagenet_classes

        self.loader = self.loadImg

        self.norm = transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                         std=(0.229, 0.224, 0.225))

        # Train Set
        self.train_transform = transforms.Compose([transforms.RandomResizedCrop(size=self.img_shape[-1]),
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.ToTensor(),
                                                   self.norm])

        self.train_set = ImageNetCustom(root='C:\Datasets\ILSVRC2012\ILSVRC2012_img_train',
                                        loader=self.loader, extensions='.jpeg', transform=self.train_transform, target_transform=None)

        self.train_data = DataLoader(dataset=self.train_set, batch_size=batch_size,
                                     shuffle=True, num_workers=num_workers, pin_memory=True)

        # Test Set
        self.test_transform = transforms.Compose([transforms.Resize(size=256),
                                                  transforms.CenterCrop(size=self.img_shape[-1]),
                                                  transforms.ToTensor(),
                                                  self.norm])

        self.test_set = ImageNetCustom(root='C:\Datasets\ILSVRC2012\ILSVRC2012_img_val',
                                       loader=self.loader, extensions='.jpeg', transform=self.test_transform, target_transform=None)

        self.test_data = DataLoader(dataset=self.test_set, batch_size=batch_size,
                                    shuffle=False, num_workers=num_workers, pin_memory=True)
        
    def loadImg(self, img_path):
        with open(img_path, 'rb') as f:
            img = Image.open(f).convert('RGB')

            return img
