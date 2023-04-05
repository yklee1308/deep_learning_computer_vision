from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader

from datasets.classes import mnist_classes


class Mnist(object):
    def __init__(self, batch_size, num_workers):
        self.img_shape = (1, 32, 32)
        self.num_classes = 10

        self.classes = mnist_classes

        self.transform = transforms.Compose([transforms.Resize(size=self.img_shape[-1]),
                                             transforms.ToTensor()])

        # Train Set
        self.train_set = MNIST(root='C:\Datasets\MNIST\MNIST_img_train',
                               transform=self.transform, train=True, download=True)
        
        self.train_data = DataLoader(dataset=self.train_set, batch_size=batch_size,
                                     shuffle=True, num_workers=num_workers, pin_memory=True)

        # Test Set
        self.test_set = MNIST(root='C:\Datasets\MNIST\MNIST_img_val',
                              transform=self.transform, train=False, download=True)
        
        self.test_data = DataLoader(dataset=self.test_set, batch_size=batch_size,
                                    shuffle=False, num_workers=num_workers, pin_memory=True)
