import torch
from torchsummary import summary

from configs import parser

from models import getModel
from datasets import getDataset
from tester import getTester


def test(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set Dataset
    dataset = getDataset(args.dataset)(input_shape=args.input_shape, batch_size=args.batch_size, num_workers=args.num_workers)
    print('Successfully loaded dataset : [Dataset] {}\n'.format(args.dataset))

    # Set Model
    model = getModel(args.model)(in_channels=dataset.input_shape[0], out_channels=(dataset.num_classes, dataset.bbox_channels)).to(device)
    print('Successfully loaded model : [Model] {}\n'.format(args.model))

    summary(model, input_size=dataset.input_shape)

    # Set Tester
    tester = getTester()(model=model, dataset=dataset, device=device, args=args)
    print('Successfully loaded tester\n')

    # Testing
    tester.test()
    print('Successfully finished testing\n')

    tester.inference()
    print('Successfully inferenced\n')


if __name__ == '__main__':
    test(parser.parse_args())
