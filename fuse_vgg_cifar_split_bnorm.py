import os
import torch
import torchvision

import torchvision.transforms as transforms

from torch.nn import CrossEntropyLoss
from torch.utils.data import ConcatDataset

from REPAIR.fuse_diff import fuse_from_cfg
from REPAIR.net_models.models import VGG
from REPAIR.fuse_cfg import BaseFuseCfg
from REPAIR.matching.weight_matching_gen import WeightMatching
from REPAIR.matching.activation_matching import ActivationMatching
from REPAIR.matching.ste_weight_matching_gen import SteMatching


def get_datasets():
    path   = os.path.dirname(os.path.abspath(__file__))

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
        batch_size=256,
        shuffle=True,
        num_workers=8)
    
    SecondHalfLoader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(mnistTrainSet, second_half),
        batch_size=256,
        shuffle=True,
        num_workers=8)
    
    ConcatLoader = torch.utils.data.DataLoader(
        ConcatDataset((torch.utils.data.Subset(mnistTrainSet, first_half), torch.utils.data.Subset(mnistTrainSet, second_half))), 
        batch_size=256,
        shuffle=True, 
        num_workers=8
    )
    
    return FirstHalfLoader, SecondHalfLoader, ConcatLoader


if __name__ == "__main__":
    loader0, loader1, loaderc = get_datasets()

    fuse_cfg = BaseFuseCfg(num_experiments=4, alpha_split=20)
    
    fuse_cfg.proj_name = "vgg_cifar_split_bnorm_log"

    vgg_cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
    fuse_cfg.models = {
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
        }
    }
    fuse_cfg.configs = {
        0: {
            "match_method": WeightMatching(
                epochs=1000,
                debug=False
            ),
            "device": "cuda"
        },
        1: {
            "match_method": WeightMatching(
                epochs=1000,
                debug=False
            ),
            "device": "cuda"
        },
        2: {
            "match_method": WeightMatching(
                epochs=1000,
                debug=False
            ),
            "device": "cuda"
        },
        3: {
            "match_method": WeightMatching(
                epochs=1000,
                debug=False
            ),
            "device": "cuda"
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
        },
        2: {
            "loader0": loader0,
            "loader1": loader1,
            "loaderc": loaderc,
        },
        3: {
            "loader0": loader0,
            "loader1": loader1,
            "loaderc": loaderc,
        }
    }
    fuse_cfg.names = {
        0: {
            "experiment_name": "fuse_mlp_cifar_split_WM_0",
            "model0_name": "vgg_cifar_split_first_bnorm_0",
            "model1_name": "vgg_cifar_split_second_bnorm_0"
        },
        1: {
            "experiment_name": "fuse_mlp_cifar_split_WM_1",
            "model0_name": "vgg_cifar_split_first_bnorm_1",
            "model1_name": "vgg_cifar_split_second_bnorm_1"
        },
        2: {
            "experiment_name": "fuse_mlp_cifar_split_WM_2",
            "model0_name": "vgg_cifar_split_first_bnorm_2",
            "model1_name": "vgg_cifar_split_second_bnorm_2"
        },
        3: {
            "experiment_name": "fuse_mlp_cifar_split_WM_3",
            "model0_name": "vgg_cifar_split_first_bnorm_3",
            "model1_name": "vgg_cifar_split_second_bnorm_3"
        }
    }
    
    fuse_cfg.root_path = os.path.dirname(os.path.abspath(__file__))

    fuse_from_cfg(fuse_cfg)