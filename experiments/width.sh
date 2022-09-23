#!/bin/bash
python train.py --num-samples=200 --model-type=ResNet18VW --width=4 --dataset=CIFAR10 --random-labels=True --experiment-name=width_4 --compute-test=False
python train.py --num-samples=200 --model-type=ResNet18VW --width=8 --dataset=CIFAR10 --random-labels=True --experiment-name=width_8 --compute-test=False
python train.py --num-samples=200 --model-type=ResNet18VW --width=16 --dataset=CIFAR10 --random-labels=True --experiment-name=width_16 --compute-test=False
python train.py --num-samples=200 --model-type=ResNet18VW --width=32 --dataset=CIFAR10 --random-labels=True --experiment-name=width_32 --compute-test=False
python train.py --num-samples=200 --model-type=ResNet18VW --width=64 --dataset=CIFAR10 --random-labels=True --experiment-name=width_64 --compute-test=False