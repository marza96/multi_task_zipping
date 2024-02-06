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

    fuse_cfg = BaseFuseCfg(num_experiments=6, alpha_split=20)
    
    fuse_cfg.proj_name = "fuse_vgg_cifar_split_bnorm_sweep"

    vgg_cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
    
    models  = dict()
    names   = dict()
    configs = dict()
    loaders = dict()
    
    for i in range(6):
        models.update(
            {
                i: {
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
                i: {
                    "match_method": WeightMatching(
                        epochs=1000,
                        debug=False
                    ),
                    "device": "cuda"
                }
            }
        )
        loaders.update(
            {
                i: {
                    "loader0": loader0,
                    "loader1": loader1,
                    "loaderc": loaderc,
                },
            }
        )
        names.update(
            {
                i: {
                    "experiment_name": "fuse_mlp_cifar_split_WM_%d" % i,
                    "model0_name": "vgg_cifar_split_first_bnorm_%d" % i,
                    "model1_name": "vgg_cifar_split_second_bnorm_%d" % i
                }
            }
        )
    
    fuse_cfg.models    = models
    fuse_cfg.configs   = configs
    fuse_cfg.loaders   = loaders
    fuse_cfg.names     = names
    fuse_cfg.root_path = os.path.dirname(os.path.abspath(__file__))

    fuse_from_cfg(fuse_cfg)