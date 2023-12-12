import os
import torch
import torchvision

import torchvision.transforms as transforms

from torch.nn import CrossEntropyLoss
from torch.utils.data import ConcatDataset

from REPAIR.fuse_diff import fuse_from_cfg
from REPAIR.net_models.mlp import VGG
from REPAIR.fuse_cfg import BaseFuseCfg
from REPAIR.matching.weight_matching import WeightMatching
from REPAIR.matching.activation_matching import ActivationMatching


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
    
    ConcatLoader = torch.utils.data.DataLoader(
        ConcatDataset((torch.utils.data.Subset(mnistTrainSet, first_half), torch.utils.data.Subset(mnistTrainSet, second_half))), 
        batch_size=128,
        shuffle=True, 
        num_workers=8
    )
    
    return FirstHalfLoader, SecondHalfLoader, ConcatLoader


if __name__ == "__main__":
    loader0, loader1, loaderc = get_datasets()

    fuse_cfg = BaseFuseCfg(num_experiments=2, alpha_split=10)

    vgg_cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
    fuse_cfg.models = {
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
        },
    }
    fuse_cfg.configs = {
        0: {
            "match_method": ActivationMatching(
                loaderc,
                epochs=1,
                device="mps"
            ),
            "device": "cuda"
        },
        1: {
            "match_method": WeightMatching(
                epochs=1000,
                debug=False
            ),
            "device": "cpu"
        }
    }
    fuse_cfg.loaders = {
        0: {
            "loader0": loader0,
            "loader1": loader1,
            "loaderc": loaderc,
        },
        1: {
            "loader0": loader0,
            "loader1": loader1,
            "loaderc": loaderc,
        }
    }
    fuse_cfg.names = {
        0: {
            "experiment_name": "fuse_mlp_cifar_split_AM",
            "model0_name": "vgg_first_cifar_split",
            "model1_name": "vgg_second_cifar_split"
        },
        1: {
            "experiment_name": "fuse_mlp_cifar_split_WM",
            "model0_name": "vgg_first_cifar_split",
            "model1_name": "vgg_second_cifar_split"
        }
    }
    
    fuse_cfg.root_path = os.path.dirname(__file__)

    fuse_from_cfg(fuse_cfg)