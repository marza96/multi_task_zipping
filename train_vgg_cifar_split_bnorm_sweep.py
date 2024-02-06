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
    
    vgg_cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
    lrs     = [0.01, 0.01, 0.025, 0.025, 0.0075, 0.0075]
    wds     = [0.005, 0.001, 0.005, 0.001, 0.005, 0.001]
    epochs  = [25, 13, 25, 13, 25, 13]
    
    train_cfg           = BaseTrainCfg(num_experiments=2 * len(lrs))
    train_cfg.proj_name = "train_vgg_cifar_split_bnorm_sweep"
    
    models  = dict()
    names   = dict()
    configs = dict()
    loaders = dict()

    for i in range(len(lrs)):
        models.update(
            {
                2 * i: {
                    "model": VGG,
                    "args": {
                        "w": 2,
                        "cfg": vgg_cfg,
                        "classes": 10,
                        "bnorm": True
                    }
                }
            }
        )
        models.update(
            {
                2 * i + 1: {
                    "model": VGG,
                    "args": {
                        "w": 2,
                        "cfg": vgg_cfg,
                        "classes": 10,
                        "bnorm": True
                    }
                }
            }
        )
        configs.update(
            {
                2 * i: {
                    "loss_fn": CrossEntropyLoss(),
                    "epochs" : epochs[i], #13
                    "device": "cuda",
                    "optimizer": {
                        "class": torch.optim.SGD,
                        "args": {
                            "lr": lrs[i],
                            "momentum": 0.9,
                            "weight_decay": wds[i]
                        }
                    }
                }
            }
        )
        configs.update(
            {
                2 * i + 1: {
                    "loss_fn": CrossEntropyLoss(),
                    "epochs" : epochs[i], #13
                    "device": "cuda",
                    "optimizer": {
                        "class": torch.optim.SGD,
                        "args": {
                            "lr": lrs[i],
                            "momentum": 0.9,
                            "weight_decay": wds[i]
                        }
                    }
                }
            }
        )
        loaders.update(
            {
                2 * i: {"train": loader0, "test": loader0_test}
            }
        )
        loaders.update(
            {
                2 * i + 1: {"train": loader1, "test": loader1_test}
            }
        )
        names.update(
            {
                2 * i: "vgg_cifar_split_first_bnorm_%d" % i
            }
        )
        names.update(
            {
                2 * i + 1: "vgg_cifar_split_second_bnorm_%d" % i
            }
        )
    
    train_cfg.models    = models
    train_cfg.configs   = configs
    train_cfg.loaders   = loaders
    train_cfg.names     = names
    train_cfg.root_path = os.path.dirname(os.path.abspath(__file__))

    train_from_cfg(train_cfg)