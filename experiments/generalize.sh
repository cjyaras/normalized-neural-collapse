#!/bin/bash
python train.py --epochs=100 --model-type=ResNet18 --dataset=CIFAR100 --normalize=True --experiment-name=generalize_resnet18_normalized --compute-test=True
python train.py --epochs=100 --model-type=ResNet18 --dataset=CIFAR100 --normalize=False --experiment-name=generalize_resnet18_regularized --compute-test=True

python train.py --epochs=100 --model-type=ResNet50 --dataset=CIFAR100 --normalize=True --experiment-name=generalize_resnet50_normalized --compute-test=True
python train.py --epochs=100 --model-type=ResNet50 --dataset=CIFAR100 --normalize=False --experiment-name=generalize_resnet50_regularized --compute-test=True