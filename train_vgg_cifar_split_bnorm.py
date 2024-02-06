import os
import torch
import torchvision

import numpy as np

import torchvision.transforms as transforms

from torch.optim import SGD
from torch.nn import CrossEntropyLoss

from REPAIR.train import train_from_cfg
from REPAIR.net_models.models import VGG
from REPAIR.train_cfg import BaseTrainCfg

#NOTE
#NOTE
#NOTE
#NOTE IMPORTANT NOTE
#     YOU FORGOT TO ADD BATCHNORM TO THE LAST LAYER
#     OF VGG (classifier) for the case when bnorm == True
def get_datasets(train=True):
    path   = os.path.dirname(os.path.abspath(__file__))

    # MEAN = [0.4906, 0.4856, 0.4508]
    # STD  = [0.2454, 0.2415, 0.2620]

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            # torchvision.transforms.Normalize(np.array(MEAN), np.array(STD))
        ]
    )
    mnistTrainSet = torchvision.datasets.CIFAR10(
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
        batch_size=256,
        shuffle=True,
        num_workers=8)
    
    SecondHalfLoader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(mnistTrainSet, second_half),
        batch_size=256,
        shuffle=True,
        num_workers=8)
    
    return FirstHalfLoader, SecondHalfLoader


if __name__ == "__main__":
    loader0, loader1 = get_datasets()
    loader0_test, loader1_test = get_datasets(train=False)
    
    train_cfg = BaseTrainCfg(num_experiments=8)
    vgg_cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
    train_cfg.proj_name = "vgg_cifar_split_bnorm"
    train_cfg.models = {
        0: {
            "model": VGG,
            "args": {
                "w": 2,
                "cfg": vgg_cfg,
                "classes": 10,
                "bnorm": True
            }
        },
        1: {
            "model": VGG,
            "args": {
                "w": 2,
                "cfg": vgg_cfg,
                "classes": 10,
                "bnorm": True
            }
        },
        2: {
            "model": VGG,
            "args": {
                "w": 2,
                "cfg": vgg_cfg,
                "classes": 10,
                "bnorm": True
            }
        },
        3: {
            "model": VGG,
            "args": {
                "w": 2,
                "cfg": vgg_cfg,
                "classes": 10,
                "bnorm": True
            }
        },
        4: {
            "model": VGG,
            "args": {
                "w": 2,
                "cfg": vgg_cfg,
                "classes": 10,
                "bnorm": True
            }
        },
        5: {
            "model": VGG,
            "args": {
                "w": 2,
                "cfg": vgg_cfg,
                "classes": 10,
                "bnorm": True
            }
        },
        6: {
            "model": VGG,
            "args": {
                "w": 2,
                "cfg": vgg_cfg,
                "classes": 10,
                "bnorm": True
            }
        },
        7: {
            "model": VGG,
            "args": {
                "w": 2,
                "cfg": vgg_cfg,
                "classes": 10,
                "bnorm": True
            }
        }
    }
    train_cfg.configs = {
        0: {
            "loss_fn": CrossEntropyLoss(),
            "epochs" : 13, #13
            "device": "cuda",
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
            "epochs": 13,  #13
            "device": "cuda",
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
            "epochs" : 20,
            "device": "cuda",
            "optimizer": {
                "class": torch.optim.SGD,
                "args": {
                    "lr": 0.1,
                    "momentum": 0.9,
                    "weight_decay": 0.0001
                }
            }
        },
        3: {
            "loss_fn": CrossEntropyLoss(),
            "epochs": 20,
            "device": "cuda",
            "optimizer": {
                "class": torch.optim.SGD,
                "args": {
                    "lr": 0.1,
                    "momentum": 0.9,
                    "weight_decay": 0.0001
                }
            }
        },
        4: {
            "loss_fn": CrossEntropyLoss(),
            "epochs" : 20,
            "device": "cuda",
            "optimizer": {
                "class": torch.optim.SGD,
                "args": {
                    "lr": 0.5,
                    "momentum": 0.9,
                    "weight_decay": 0.0001
                }
            }
        },
        5: {
            "loss_fn": CrossEntropyLoss(),
            "epochs": 20,
            "device": "cuda",
            "optimizer": {
                "class": torch.optim.SGD,
                "args": {
                    "lr": 0.5,
                    "momentum": 0.9,
                    "weight_decay": 0.0001
                }
            }
        },
        6: {
            "loss_fn": CrossEntropyLoss(),
            "epochs" : 20,
            "device": "cuda",
            "optimizer": {
                "class": torch.optim.SGD,
                "args": {
                    "lr": 0.05,
                    "momentum": 0.9,
                    "weight_decay": 0.0001
                }
            }
        },
        7: {
            "loss_fn": CrossEntropyLoss(),
            "epochs": 20,
            "device": "cuda",
            "optimizer": {
                "class": torch.optim.SGD,
                "args": {
                    "lr": 0.05,
                    "momentum": 0.9,
                    "weight_decay": 0.0001
                }
            }
        },
    }
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
        0: "vgg_cifar_split_first_bnorm_0",
        1: "vgg_cifar_split_second_bnorm_0",
        2: "vgg_cifar_split_first_bnorm_1",
        3: "vgg_cifar_split_second_bnorm_1",
        4: "vgg_cifar_split_first_bnorm_2",
        5: "vgg_cifar_split_second_bnorm_2",
        6: "vgg_cifar_split_first_bnorm_3",
        7: "vgg_cifar_split_second_bnorm_3"
    }
    train_cfg.root_path = os.path.dirname(os.path.abspath(__file__))

    train_from_cfg(train_cfg)