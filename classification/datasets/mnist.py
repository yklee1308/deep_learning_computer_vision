from PIL import Image

from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder
from torchvision import transforms

from datasets.classes.mnist_classes import mnist_classes


class MNISTCustom(DatasetFolder):
    def __init__(self, root, loader, extensions, transform, target_transform):
        super(MNISTCustom, self).__init__(root=root, loader=loader, extensions=extensions, transform=transform, target_transform=target_transform)
    
    def __getitem__(self, idx):
        x, y = self.samples[idx], int(self.targets[idx])

        img_path = x[0]
        img = self.loader(img_path=img_path)
        x = self.transform(img)

        if self.target_transform != None:
            y = self.target_transform(y)
            
        return x, y, img_path

class MNIST(object):
    def __init__(self, img_shape, batch_size, num_workers):
        self.img_shape = img_shape
        self.num_classes = 10

        self.classes = mnist_classes

        self.loader = self.loadImage

        self.transform = transforms.Compose([transforms.Resize(size=self.img_shape[-1]),
                                             transforms.ToTensor()])

        # Train Set
        self.train_set = MNISTCustom(root='C:\Datasets\MNIST\MNIST_img_train',
                                     loader=self.loader, extensions='.jpg', transform=self.transform, target_transform=None)
        
        self.train_data = DataLoader(dataset=self.train_set, batch_size=batch_size,
                                     shuffle=True, num_workers=num_workers, pin_memory=True)

        # Test Set
        self.test_set = MNISTCustom(root='C:\Datasets\MNIST\MNIST_img_val',
                                    loader=self.loader, extensions='.jpg', transform=self.transform, target_transform=None)
        
        self.test_data = DataLoader(dataset=self.test_set, batch_size=batch_size,
                                    shuffle=False, num_workers=num_workers, pin_memory=True)
        
        # Inference Data
        self.inference_data = DataLoader(dataset=self.test_set, batch_size=1,
                                         shuffle=True, num_workers=num_workers, pin_memory=True)
        
    def loadImage(self, img_path):
        with open(img_path, 'rb') as f:
            img = Image.open(f).convert(mode='L')

            return img
