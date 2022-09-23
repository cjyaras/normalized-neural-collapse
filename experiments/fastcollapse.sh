#!/bin/bash
python train.py --num-samples=30 --epochs=100 --model-type=ResNet18 --dataset=CIFAR100 --normalize=True --experiment-name=fast_resnet18_normalized
python train.py --num-samples=30 --epochs=100 --model-type=ResNet18 --dataset=CIFAR100 --normalize=False --experiment-name=fast_resnet18_regularized

python train.py --num-samples=30 --epochs=100 --model-type=ResNet50 --dataset=CIFAR100 --normalize=True --experiment-name=fast_resnet50_normalized
python train.py --num-samples=30 --epochs=100 --model-type=ResNet50 --dataset=CIFAR100 --normalize=False --experiment-name=fast_resnet50_regularized