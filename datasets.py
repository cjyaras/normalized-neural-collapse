import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, MNIST, CIFAR100
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

def create_dataset(dataset_name, data_dir, batch_size, sample_size, num_classes, random_labels=False):

   if dataset_name == 'CIFAR10':
      train_dataset = CIFAR10(root=data_dir, train=True, download=True, transform=transforms.Compose([
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
      ]))

      test_dataset = CIFAR10(root=data_dir, train=False, download=True, transform=transforms.Compose([
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
      ]))

   elif dataset_name == 'CIFAR100':
      train_dataset = CIFAR100(root=data_dir, train=True, download=True, transform=transforms.Compose([
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
      ]))

      test_dataset = CIFAR100(root=data_dir, train=False, download=True, transform=transforms.Compose([
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
      ]))

   elif dataset_name == 'MNIST':
      train_dataset = MNIST(root=data_dir, train=True, download=True, transform=transforms.Compose([
         transforms.Grayscale(3),
         transforms.Resize(32),
         transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))
      ]))

      test_dataset = MNIST(root=data_dir, train=False, download=True, transform=transforms.Compose([
         transforms.Grayscale(3),
         transforms.Resize(32),
         transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))
      ]))

   else:
      raise ValueError

   if random_labels:
      train_dataset.targets = np.random.permutation(train_dataset.targets).tolist()
      test_dataset.targets = np.random.permutation(test_dataset.targets).tolist()

   if sample_size is not None:
      total_sample_size = num_classes * sample_size
      cnt_dict = dict()
      total_cnt = 0
      indices = []
      for i in range(len(train_dataset)):
         if total_cnt == total_sample_size:
            break

         label = train_dataset[i][1]
         if label not in cnt_dict:
            cnt_dict[label] = 1
            total_cnt += 1
            indices.append(i)
         else:
            if cnt_dict[label] == sample_size:
               continue
            else:
               cnt_dict[label] += 1
               total_cnt += 1
               indices.append(i)

      train_indices = torch.tensor(indices)
      train_dataloader = DataLoader(
         train_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_indices), shuffle=False
      )
      
   else:

      train_dataloader = DataLoader(
         train_dataset, batch_size=batch_size, shuffle=False
      )

   test_dataloader = DataLoader(
      test_dataset, batch_size=batch_size, shuffle=False
   )

   return train_dataset, test_dataset, train_dataloader, test_dataloader
