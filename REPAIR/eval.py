from REPAIR import eval
from net_models.mlp import MLP

import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from torch.utils.data import ConcatDataset, DataLoader

import tqdm
import copy
import torch
import torchvision

def save_model(model, i):
    sd = model.state_dict()
    torch.save(sd, i)


def load_model(model, i):
    sd = torch.load(i)
    model.load_state_dict(sd)

if __name__ == "__main__":
    h          = 128
    layers     = 5
    device     = torch.device("cuda")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    FashionMNISTTrainSet = torchvision.datasets.FashionMNIST(
        root='./data', 
        train=True,
        download=True, 
        transform=transform
    )
    FashionMNISTTrainLoader = torch.utils.data.DataLoader(
        FashionMNISTTrainSet, 
        batch_size=128,
        shuffle=True, 
        num_workers=8
    )
    MNISTTrainSet = torchvision.datasets.MNIST(
        root='./data', 
        train=True,
        download=True, 
        transform=transform
    )
    MNISTTrainLoader = torch.utils.data.DataLoader(
        MNISTTrainSet, 
        batch_size=128,
        shuffle=True, 
        num_workers=8
    )

    ConcatTrainLoader = torch.utils.data.DataLoader(
        ConcatDataset((FashionMNISTTrainSet, MNISTTrainSet)), 
        batch_size=128,
        shuffle=True, 
        num_workers=8
    )

    model0 = MLP(h, layers).to(device)
    load_model(model0, "merged.pt")
    eval_val = eval.evaluate_acc(model0, loader=FashionMNISTTrainLoader, device=device)
    print("JOINT ACC", eval_val)