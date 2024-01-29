import os
import torch
import torchvision

import torchvision.transforms as transforms

from torch.optim import SGD
from torch.nn import CrossEntropyLoss

from REPAIR.train import train_from_cfg
from REPAIR.net_models.models import MLP
from REPAIR.train_cfg import BaseTrainCfg


def get_datasets():
    path   = os.path.dirname(os.path.abspath(__file__))

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    mnistTrainSet = torchvision.datasets.FashionMNIST(
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

    train_cfg = BaseTrainCfg(num_experiments=6)

    train_cfg.models = {
        0: {
            "model": MLP,
            "args": {
                "layers": 5,
                "channels": 512,
                "classes": 10,
                "bnorm": True
            }
        },
        1: {
            "model": MLP,
            "args": {
                "layers": 5,
                "channels": 512,
                "classes": 10,
                "bnorm": True
            }
        },
        2: {
            "model": MLP,
            "args": {
                "layers": 5,
                "channels": 512,
                "classes": 10,
                "bnorm": True
            }
        },
        3: {
            "model": MLP,
            "args": {
                "layers": 5,
                "channels": 512,
                "classes": 10,
                "bnorm": True
            }
        },
        4: {
            "model": MLP,
            "args": {
                "layers": 5,
                "channels": 512,
                "classes": 10,
                "bnorm": True
            }
        },
        5: {
            "model": MLP,
            "args": {
                "layers": 5,
                "channels": 512,
                "classes": 10,
                "bnorm": True
            }
        }
    }
    train_cfg.configs = {
        0: {
            "loss_fn": CrossEntropyLoss(),
            "epochs" : 14,
            "device": "mps",
            "optimizer": {
                "class": torch.optim.SGD,
                "args": {
                    "lr": 0.01,
                    "momentum": 0.9,
                    "weight_decay": 0.005
                }
            }
        },
        1: {
            "loss_fn": CrossEntropyLoss(),
            "epochs": 5,
            "device": "mps",
            "optimizer": {
                "class": torch.optim.SGD,
                "args": {
                    "lr": 0.01,
                    "momentum": 0.9,
                    "weight_decay": 0.005
                }
            }
        },
        2: {
            "loss_fn": CrossEntropyLoss(),
            "epochs" : 15,
            "device": "mps",
            "optimizer": {
                "class": torch.optim.Adam,
                "args": {
                    "lr": 0.001,
                    "weight_decay": 0.005
                }
            }
        },
        3: {
            "loss_fn": CrossEntropyLoss(),
            "epochs": 15,
            "device": "mps",
            "optimizer": {
                "class": torch.optim.Adam,
                "args": {
                    "lr": 0.001,
                    "weight_decay": 0.005
                }
            }
        },
        4: {
            "loss_fn": CrossEntropyLoss(),
            "epochs" : 35,
            "device": "mps",
            "optimizer": {
                "class": torch.optim.Adam,
                "args": {
                    "lr": 0.0001,
                    "weight_decay": 0.005
                }
            }
        },
        5: {
            "loss_fn": CrossEntropyLoss(),
            "epochs": 35,
            "device": "mps",
            "optimizer": {
                "class": torch.optim.Adam,
                "args": {
                    "lr": 0.0001,
                    "weight_decay": 0.005
                }
            }
        }
    }
    train_cfg.loaders = {
        0: loader0,
        1: loader1,
        2: loader0,
        3: loader1,
        4: loader0,
        5: loader1
    }
    train_cfg.names = {
        0: "mlp_first_fmnist_bnorm_0",
        1: "mlp_second_fmnist_bnorm_1",
        2: "mlp_first_fmnist_bnorm_2",
        3: "mlp_second_fmnist_bnorm_3",
        4: "mlp_first_fmnist_bnorm_4",
        5: "mlp_second_fmnist_bnorm_5",
    }
    train_cfg.root_path = os.path.dirname(os.path.abspath(__file__))

    train_from_cfg(train_cfg)