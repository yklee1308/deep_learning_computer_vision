import argparse


parser = argparse.ArgumentParser(description='detection settings')

# Model
parser.add_argument('--model', type=str, default='R-CNN',  help='model')

# Dataset
parser.add_argument('--dataset', type=str, default='VOC2007', help='dataset')

# Input
parser.add_argument('--img_shape', type=tuple, default=(3, 227, 227), help='img_shape')

# Processing
parser.add_argument('--num_regions', type=int, default=128, help='num_regions')
parser.add_argument('--positive_ratio', type=float, default=0.25, help='positive_ratio')
parser.add_argument('--iou_th', type=float, default=0.5, help='iou_th')

# Training
parser.add_argument('--loss_function', type=tuple, default=('BCE', 'MSE'), help='loss_function')
parser.add_argument('--loss_weight', type=tuple, default=(1, 1), help='loss_weight')

parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight_decay')
parser.add_argument('--learning_rate', type=float, default=0.01, help='learning_rate')

parser.add_argument('--epochs', type=int, default=100, help='epochs')
parser.add_argument('--batch_size', type=int, default=1, help='batch_size')

parser.add_argument('--resume_training', type=bool, default=False, help='resume_training')

# System
parser.add_argument('--num_workers', type=int, default=16, help='num_workers')
