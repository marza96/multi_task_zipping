#!/bin/bash

DEVICE="cuda"
D0="FashionMNIST"
D1="MNIST"

python REPAIR/train.py --device $DEVICE --dataset0 $D0 --dataset1 $D1