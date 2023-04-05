import argparse


parser = argparse.ArgumentParser(description='classification settings')

# Dataset
parser.add_argument('--model', type=str, default='LeNet',  help='model')

# Model
parser.add_argument('--dataset', type=str, default='MNIST', help='dataset')

# Training
parser.add_argument('--loss_function', type=str, default='cross_entropy', help='loss_function')
parser.add_argument('--optimizer', type=str, default='stochastic_gradient_descent', help='optimizer')

parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay')
parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')

parser.add_argument('--epochs', type=int, default=100, help='epochs')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')

# Environment
parser.add_argument('--num_workers', type=int, default=16, help='num_workers')

parser.add_argument('--resume_training', type=bool, default=False, help='resume_training')
