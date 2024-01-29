import os
import torch
import torchvision

import torchvision.transforms as transforms

from torch.utils.data import ConcatDataset

from REPAIR.fuse_diff import fuse_from_cfg
from REPAIR.net_models.models import MLP
from REPAIR.fuse_cfg import BaseFuseCfg
from REPAIR.matching.weight_matching import WeightMatching
from REPAIR.matching.activation_matching import ActivationMatching
from REPAIR.matching.ste_weight_matching import SteMatching


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
    
    ConcatLoader = torch.utils.data.DataLoader(
        ConcatDataset((torch.utils.data.Subset(mnistTrainSet, first_half), torch.utils.data.Subset(mnistTrainSet, second_half))), 
        batch_size=512,
        shuffle=True, 
        num_workers=8
    )
    
    return FirstHalfLoader, SecondHalfLoader, ConcatLoader


if __name__ == "__main__":
    loader0, loader1, loaderc = get_datasets()

    fuse_cfg = BaseFuseCfg(num_experiments=3, alpha_split=10)

    fuse_cfg.models = {
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
    }
    fuse_cfg.configs = {
        0: {
            "match_method": ActivationMatching(
                loaderc,
                epochs=1,
                device="cuda"
            ),
            "device": "cuda"
        },
        1: {
            "match_method": WeightMatching(
                epochs=1000,
                debug=False
            ),
            "device": "cpu"
        },
        2: {
            "match_method": SteMatching(
                torch.nn.functional.cross_entropy,
                loaderc,
                0.25,
                WeightMatching(
                    epochs=1000,
                    ret_perms=True
                ),
                epochs=20,
                device="cuda"
            ),
            "device": "cuda"
        },
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
        }
    }
    fuse_cfg.names = {
        0: {
            "experiment_name": "fuse_mlp_fash_mnist_split_AM",
            "model0_name": "mlp_first_fmnist_bnorm_1",
            "model1_name": "mlp_second_fmnist_bnorm_1"
        },
        1: {
            "experiment_name": "fuse_mlp_fash_mnist_split_WM",
            "model0_name": "mlp_first_fmnist_bnorm_1",
            "model1_name": "mlp_second_fmnist_bnorm_1"
        },
        2: {
            "experiment_name": "fuse_mlp_fash_mnist_split_STE",
            "model0_name": "mlp_first_fmnist_bnorm_1",
            "model1_name": "mlp_second_fmnist_bnorm_1"
        }
    }
    fuse_cfg.root_path = os.path.dirname(os.path.abspath(__file__))

    fuse_from_cfg(fuse_cfg)