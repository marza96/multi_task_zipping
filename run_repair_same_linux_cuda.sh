#!/bin/bash

DEVICE="cuda"
M0="fash_mnist_mlp_e50_l5_h128_v1_cuda.pt"
M1="fash_mnist_mlp_e50_l5_h128_v2_cuda.pt"

python REPAIR/fuse_same.py --device $DEVICE --model0_path $M0 --model1_path $M1