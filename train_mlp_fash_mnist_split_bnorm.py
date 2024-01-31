import os
import torch
import torchvision

import torchvision.transforms as transforms

from torch.optim import SGD
from torch.nn import CrossEntropyLoss

from REPAIR.train import train_from_cfg
from REPAIR.net_models.models import MLP
from REPAIR.train_cfg import BaseTrainCfg


def get_datasets(train=True):
    path   = os.path.dirname(os.path.abspath(__file__))

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    mnistTrainSet = torchvision.datasets.FashionMNIST(
        root=path + '/data', 
        train=train,
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
    loader0_test, loader1_test = get_datasets(train=False)

    train_cfg = BaseTrainCfg(num_experiments=8)

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
        },
        6: {
            "model": MLP,
            "args": {
                "layers": 5,
                "channels": 512,
                "classes": 10,
                "bnorm": True
            }
        },
        7: {
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
            "epochs" : 65,
            "device": "cuda",
            "optimizer": {
                "class": torch.optim.Adam,
                "args": {
                    "lr": 0.0001,
                    "weight_decay": 0.005
                }
            }
        },
        1: {
            "loss_fn": CrossEntropyLoss(),
            "epochs": 20,
            "device": "cuda",
            "optimizer": {
                "class": torch.optim.Adam,
                "args": {
                    "lr": 0.0001,
                    "weight_decay": 0.005
                }
            }
        },
        2: {
            "loss_fn": CrossEntropyLoss(),
            "epochs" : 25,
            "device": "cuda",
            "optimizer": {
                "class": torch.optim.Adam,
                "args": {
                    "lr": 0.0005,
                    "weight_decay": 0.005
                }
            }
        },
        3: {
            "loss_fn": CrossEntropyLoss(),
            "epochs": 25,
            "device": "cuda",
            "optimizer": {
                "class": torch.optim.Adam,
                "args": {
                    "lr": 0.0005,
                    "weight_decay": 0.005
                }
            }
        },
        4: {
            "loss_fn": CrossEntropyLoss(),
            "epochs" : 35,
            "device": "cuda",
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
            "device": "cuda",
            "optimizer": {
                "class": torch.optim.Adam,
                "args": {
                    "lr": 0.0001,
                    "weight_decay": 0.005
                }
            }
        },
        6: {
            "loss_fn": CrossEntropyLoss(),
            "epochs" : 40,
            "device": "cuda",
            "optimizer": {
                "class": torch.optim.Adam,
                "args": {
                    "lr": 0.0001,
                    "weight_decay": 0.005
                }
            }
        },
        7: {
            "loss_fn": CrossEntropyLoss(),
            "epochs": 40,
            "device": "cuda",
            "optimizer": {
                "class": torch.optim.Adam,
                "args": {
                    "lr": 0.0001,
                    "weight_decay": 0.005
                }
            }
        }
    }
    
    train_cfg.proj_name = "mlp_fmnist_bnorm"
    train_cfg.loaders = {
        0: {"train": loader0, "test": loader0_test},
        1: {"train": loader1, "test": loader1_test},
        2: {"train": loader0, "test": loader0_test},
        3: {"train": loader1, "test": loader1_test},
        4: {"train": loader0, "test": loader0_test},
        5: {"train": loader1, "test": loader1_test},
        6: {"train": loader0, "test": loader0_test},
        7: {"train": loader1, "test": loader1_test},
    }
    train_cfg.names = {
        0: "mlp_first_fmnist_bnorm_0",
        1: "mlp_second_fmnist_bnorm_0",
        2: "mlp_first_fmnist_bnorm_1",
        3: "mlp_second_fmnist_bnorm_1",
        4: "mlp_first_fmnist_bnorm_2",
        5: "mlp_second_fmnist_bnorm_2",
        6: "mlp_first_fmnist_bnorm_3",
        7: "mlp_second_fmnist_bnorm_3",
    }
    train_cfg.root_path = os.path.dirname(os.path.abspath(__file__))

    train_from_cfg(train_cfg)