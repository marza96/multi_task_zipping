#!/bin/bash

DEVICE="mps"
M0="fash_mnist_mlp_e50_l5_h128_v1_mps.pt"
M1="fash_mnist_mlp_e50_l5_h128_v2_mps.pt"

python REPAIR/fuse_same.py --device $DEVICE --model0_path $M0 --model1_path $M1