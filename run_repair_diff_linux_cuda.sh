#!/bin/bash

DEVICE="cuda"
M0="fash_mnist_mlp_e50_l3_h512_v1_mps.pt"
M1="mnist_mlp_e50_l3_h512_v2_mps.pt"

python REPAIR/fuse_diff.py --device $DEVICE --model0_path $M0 --model1_path $M1