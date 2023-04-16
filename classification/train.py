import torch
from torchsummary import summary

from configs import parser

from models import getModel
from datasets import getDataset
from trainer import getTrainer


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set Dataset
    dataset = getDataset(args.dataset)(img_shape=args.img_shape, batch_size=args.batch_size, num_workers=args.num_workers)
    print('Successfully loaded dataset : [Dataset] {}\n'.format(args.dataset))

    # Set Model
    model = getModel(args.model)(in_channels=dataset.img_shape[0], out_channels=dataset.num_classes).to(device)
    print('Successfully loaded model : [Model] {}\n'.format(args.model))

    summary(model, input_size=dataset.img_shape)

    # Set Trainer
    trainer = getTrainer()(model=model, dataset=dataset, device=device, args=args)
    print('Successfully loaded trainer : [Loss Function] {}, [Optimizer] {}\n'.format(args.loss_function, args.optimizer))

    # Training
    trainer.train()
    print('Successfully finished training')

    trainer.saveModel()
    print('Successfully saved model')


if __name__ == '__main__':
    train(parser.parse_args())
