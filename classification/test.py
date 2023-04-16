import torch
from torchsummary import summary

from configs import parser

from models import getModel
from datasets import getDataset
from tester import getTester


def test(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set Dataset
    dataset = getDataset(args.dataset)(img_shape=args.img_shape, batch_size=args.batch_size, num_workers=args.num_workers)
    print('Successfully loaded dataset : [Dataset] {}\n'.format(args.dataset))

    # Set Model
    model = getModel(args.model)(in_channels=dataset.img_shape[0], out_channels=dataset.num_classes).to(device)
    print('Successfully loaded model : [Model] {}\n'.format(args.model))

    summary(model, input_size=dataset.img_shape)

    # Set Tester
    tester = getTester()(model=model, dataset=dataset, device=device, args=args)
    print('Successfully loaded tester')

    # Testing
    tester.test()
    print('Successfully finished testing')

    tester.inference()
    print('Successfully inferenced')


if __name__ == '__main__':
    test(parser.parse_args())
