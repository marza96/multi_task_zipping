import os
import torch
import torchvision

import torchvision.transforms as transforms

from torch.optim import SGD
from torch.nn import CrossEntropyLoss

from REPAIR.train import train_from_cfg
from REPAIR.net_models.models import VGG
from REPAIR.train_cfg import BaseTrainCfg


def get_datasets():
    path   = os.path.dirname(__file__)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    mnistTrainSet = torchvision.datasets.CIFAR10(
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

    vgg_cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
    train_cfg.models = {
        0: {
            "model": VGG,
            "args": {
                "w": 1,
                "cfg": vgg_cfg,
                "classes": 10,
            }
        },
        1: {
            "model": VGG,
            "args": {
                "w": 1,
                "cfg": vgg_cfg,
                "classes": 10,
            }
        }
    }
    train_cfg.configs = {
        0: {
            "loss_fn": CrossEntropyLoss(),
            "epochs" : 50,
            "device": "cuda",
            "optimizer": {
                "class": SGD,
                "args": {
                    "lr": 0.05,
                    "momentum": 0.9
                }
            }
        },
        1: {
            "loss_fn": CrossEntropyLoss(),
            "epochs": 30,
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
        0: "vgg_cifar_split_first",
        1: "vgg_cifar_split_second"
    }
    train_cfg.root_path = os.path.dirname(__file__)

    train_from_cfg(train_cfg)