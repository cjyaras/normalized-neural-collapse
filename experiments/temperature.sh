#!/bin/bash
python train.py --model-type=ResNet18 --dataset=CIFAR10 --tau=0.001 --experiment-name=temp_0.001 --compute-test=False
python train.py --model-type=ResNet18 --dataset=CIFAR10 --tau=0.01 --experiment-name=temp_0.01 --compute-test=False
python train.py --model-type=ResNet18 --dataset=CIFAR10 --tau=0.1 --experiment-name=temp_0.1 --compute-test=False
python train.py --model-type=ResNet18 --dataset=CIFAR10 --tau=0.5 --experiment-name=temp_0.5 --compute-test=False
python train.py --model-type=ResNet18 --dataset=CIFAR10 --tau=1 --experiment-name=temp_1 --compute-test=False
python train.py --model-type=ResNet18 --dataset=CIFAR10 --tau=5 --experiment-name=temp_5 --compute-test=False
python train.py --model-type=ResNet18 --dataset=CIFAR10 --tau=10 --experiment-name=temp_10 --compute-test=False
python train.py --model-type=ResNet18 --dataset=CIFAR10 --tau=100 --experiment-name=temp_100 --compute-test=False