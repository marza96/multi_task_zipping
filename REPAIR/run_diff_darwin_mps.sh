#!/bin/bash

DEVICE="mps"
M0="pt_models/fash_mnist_mlp_e50_l5_h128_v2.pt"
M1="pt_models/mnist_mlp_e50_l5_h128_v1.pt"

python fuse_diff.py --device $DEVICE --model0_path $M0 --model1_path $M1