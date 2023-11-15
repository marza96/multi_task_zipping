#!/bin/bash

DEVICE="cuda"
M0="fash_mnist_mlp_e50_l5_h128_v2.pt"
M1="mnist_mlp_e50_l5_h128_v1.pt"

python REPAIR/fuse_diff.py --device $DEVICE --model0_path $M0 --model1_path $M1