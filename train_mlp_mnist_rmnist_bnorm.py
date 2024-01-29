import os
import copy
import torch
import torchvision

import numpy as np

import torchvision.transforms as transforms

from torch.optim import SGD
from torch.nn import CrossEntropyLoss

from REPAIR.train import train_from_cfg
from REPAIR.net_models.models import MLP
from REPAIR.train_cfg import BaseTrainCfg

from REPAIR.net_models.models import LayerWrapper, LayerWrapper2D

from torchvision.transforms.functional import rotate


def rot_img(tensor):
    return rotate(tensor, 90.0)


def get_datasets():
    path   = os.path.dirname(os.path.abspath(__file__))

    MEAN = 0.1305
    STD  = 0.3071

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            torchvision.transforms.Normalize(np.array(MEAN), np.array(STD))
        ]
    )
    mnistTrainSet = torchvision.datasets.MNIST(
        root=path + '/data', 
        train=True,
        download=True, 
        transform=transform
    )

    fashMnistTrainSet = torchvision.datasets.MNIST(
        root=path + '/data', 
        train=True,
        download=True, 
        transform=transforms.Compose(
        [
            transforms.ToTensor(),
            torchvision.transforms.Normalize(np.array(MEAN), np.array(STD)),
            rot_img
        ]
        )
    )

    first_half = [
        idx for idx, target in enumerate(fashMnistTrainSet.targets) 
        if target in [5, 6, 7, 8, 9]
    ]

    second_half = [
        idx for idx, target in enumerate(mnistTrainSet.targets) 
        if target in [0, 1, 2, 3, 4]
    ]  

    FirstHalfLoader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(fashMnistTrainSet, first_half),
        batch_size=512,
        shuffle=True,
        num_workers=8)
    
    SecondHalfLoader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(mnistTrainSet, second_half),
        batch_size=512,
        shuffle=True,
        num_workers=8)
    
    return FirstHalfLoader, SecondHalfLoader


if __name__ == "__main__":
    loader0, loader1 = get_datasets()

    train_cfg = BaseTrainCfg(num_experiments=2)

    train_cfg.models = {
        0: {
            "model": MLP,
            "args": {
                "layers": 5,
                "channels": 128,
                "classes": 10,
                "bnorm": True
            }
        },
        1: {
            "model": MLP,
            "args": {
                "layers": 5,
                "channels": 128,
                "classes": 10,
                "bnorm": True
            }
        }
    }
    train_cfg.configs = {
        0: {
            "loss_fn": CrossEntropyLoss(),
            "epochs" : 35,
            "device": "cuda",
            "optimizer": {
                "class": torch.optim.Adam,
                "args": {
                    "lr": 0.01,
                }
            }
        },
        1: {
            "loss_fn": CrossEntropyLoss(),
            "epochs": 35,
            "device": "cuda",
            "optimizer": {
                "class": torch.optim.Adam,
                "args": {
                    "lr": 0.01,
                }
            }
        }
    }
    train_cfg.loaders = {
        0: loader0,
        1: loader1
    }
    train_cfg.names = {
        0: "mlp_first_mnist_rmnist_bnorm",
        1: "mlp_second_mnist_rmnist_bnorm"
    }
    train_cfg.root_path = os.path.dirname(os.path.abspath(__file__))

    train_from_cfg(train_cfg)