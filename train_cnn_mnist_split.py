import os
import torch
import torchvision

import torchvision.transforms as transforms

from torch.optim import SGD
from torch.nn import CrossEntropyLoss

from REPAIR.train import train_from_cfg
from REPAIR.net_models.mlp import MLP
from REPAIR.net_models.mlp import CNN
from REPAIR.train_cfg import BaseTrainCfg


def get_datasets():
    path   = os.path.dirname(__file__)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    mnistTrainSet = torchvision.datasets.MNIST(
        root=path + '/data', 
        train=True,
        download=True, 
        transform=transform
    )

    first_half = [
        idx for idx, target in enumerate(mnistTrainSet.targets) 
        if target in [0, 1, 2, 3, 4]
    ]

    second_half = [
        idx for idx, target in enumerate(mnistTrainSet.targets) 
        if target in [5, 6, 7, 8, 9]
    ]  

    FirstHalfLoader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(mnistTrainSet, first_half),
        batch_size=128,
        shuffle=True,
        num_workers=8)
    
    SecondHalfLoader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(mnistTrainSet, second_half),
        batch_size=128,
        shuffle=True,
        num_workers=8)
    
    return FirstHalfLoader, SecondHalfLoader


if __name__ == "__main__":
    loader0, loader1 = get_datasets()

    train_cfg = BaseTrainCfg(num_experiments=2)

    train_cfg.models = {
        0: {
            "model": CNN,
            "args": {
                "layers": 5,
                "channels": 128,
                "classes": 10,
            }
        },
        1: {
            "model": CNN,
            "args": {
                "layers": 5,
                "channels": 128,
                "classes": 10,
            }
        }
    }
    train_cfg.configs = {
        0: {
            "loss_fn": CrossEntropyLoss(),
            "epochs" : 20,
            "device": "cuda",
            "optimizer": {
                "class": SGD,
                "args": {
                    "lr": 0.01,
                    "momentum": 0.9
                }
            }
        },
        1: {
            "loss_fn": CrossEntropyLoss(),
            "epochs": 20,
            "device": "cuda",
            "optimizer": {
                "class": SGD,
                "args": {
                    "lr": 0.01,
                    "momentum": 0.9
                }
            }
        }
    }
    train_cfg.loaders = {
        0: loader0,
        1: loader1
    }
    train_cfg.names = {
        0: "cnn_first",
        1: "cnn_second"
    }
    train_cfg.root_path = os.path.dirname(__file__)

    train_from_cfg(train_cfg)