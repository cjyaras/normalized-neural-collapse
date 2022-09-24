import click
from typing import Optional
from datetime import datetime

from utils import set_seed
from generalized_model import GeneralizedModel
from datasets import create_dataset

@click.command()
@click.option('--num-samples', type=int, help='Number of samples (per class) used for training.')
@click.option('--model-type', type=click.Choice(['ResNet18', 'ResNet50', 'ResNet18VW']), required=True)
@click.option('--width', type=int, default=64, help='Width of ResNet. Only used if using variable width network.')
@click.option('--dataset', type=click.Choice(['CIFAR10', 'CIFAR100']), required=True)
@click.option('--normalize', type=bool, default=True)
@click.option('--weight_decay', type=float, default=1e-4, help='Weight/feature decay. Only used if using unnormalized network.')
@click.option('--tau', type=float, default=1.0, help='Temperature parameter.')
@click.option('--epochs', type=int, default=200, help='Number of training epochs.')
@click.option('--batch-size', type=int, default=128, help='Batch size.')
@click.option('--lr', type=float, default=0.05, help='Initial learning rate.')
@click.option('--momentum', type=float, default=0.9, help='Momentum for SGD.')
@click.option('--lr-decay', type=float, default=0.1, help='LR decay rate.')
@click.option('--lr-decay-period', type=int, default=40, help='Period of LR decay.')
@click.option('--save-epoch', type=int, default=5, help='Frequency (in epochs) of saving weights.')
@click.option('--save-dir', type=str, default='weights', help='Relative directory path for model weights.')
@click.option('--experiment-name', type=str, default=datetime.now().strftime('%m-%d-%Y-%H_%M_%S'), help='Experiment name.')
@click.option('--random-labels', type=bool, default=False, help='Use random labels.')
@click.option('--compute-test', type=bool, help='Compute test accuracy and NC metrics.', required=True)
@click.option('--device', type=click.Choice(['cpu', 'gpu']), default='cpu')
def train(**args):
   set_seed(0)
   args['num_classes'] = 10 if args['dataset'] == 'CIFAR10' else 100
   _, _, train_dataloader, test_dataloader = create_dataset(
      dataset_name=args['dataset'],
      data_dir='datasets',
      batch_size=args['batch_size'],
      sample_size=args['num_samples'],
      num_classes=args['num_classes'],
      random_labels=args['random_labels']
   )
   model = GeneralizedModel(args, save_args=True)

   if args['compute_test']:
      model.train(train_dataloader, test_dataloader)
   else:
      model.train(train_dataloader)
   
if __name__ == '__main__':
   train()