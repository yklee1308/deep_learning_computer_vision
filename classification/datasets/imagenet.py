from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader

from datasets.classes import imagenet_classes


class ImageNet(object):
    def __init__(self, batch_size, num_workers):
        self.img_shape = (3, 227, 227)
        self.num_classes = 1000

        self.classes = imagenet_classes

        self.norm = transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                         std=(0.229, 0.224, 0.225))

        # Train Set
        self.train_transform = transforms.Compose([transforms.RandomResizedCrop(size=self.img_shape[-1]),
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.ToTensor(),
                                                   self.norm])

        self.train_set = ImageFolder(root='C:\Datasets\ILSVRC2012\ILSVRC2012_img_train',
                                     transform=self.train_transform)

        self.train_data = DataLoader(dataset=self.train_set, batch_size=batch_size,
                                     shuffle=True, num_workers=num_workers, pin_memory=True)

        # Test Set
        self.test_transform = transforms.Compose([transforms.Resize(size=256),
                                                  transforms.CenterCrop(size=self.img_shape[-1]),
                                                  transforms.ToTensor(),
                                                  self.norm])

        self.test_set = ImageFolder(root='C:\Datasets\ILSVRC2012\ILSVRC2012_img_val',
                                    transform=self.test_transform)

        self.test_data = DataLoader(dataset=self.test_set, batch_size=batch_size,
                                    shuffle=False, num_workers=num_workers, pin_memory=True)
