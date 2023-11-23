import os
from typing import Any
import torch
import torchvision

import torchvision.transforms as transforms

from torch.optim import SGD
from torch.nn import CrossEntropyLoss

from REPAIR.train import train_from_cfg
from REPAIR.net_models.mlp import VGG
from REPAIR.train_cfg import BaseTrainCfg
from REPAIR.util import load_model


def get_datasets():
    path   = os.path.dirname(__file__)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    CIFAR10TrainSet = torchvision.datasets.CIFAR10(
        root=path + '/data', 
        train=True,
        download=True, 
        transform=transform
    )

    CIFAR10Loader = torch.utils.data.DataLoader(
        CIFAR10TrainSet,
        batch_size=128,
        shuffle=True,
        num_workers=8)
    
    return CIFAR10Loader


class ModelModifier:
    def __init__(self, model_src):
        self.model_src = model_src

    def __call__(self, model_dst):
        model_dst.load_state_dict(
            self.model_src.state_dict()
        )

        for param in model_dst.parameters():
            param.requires_grad = False

        for name, param in model_dst.named_parameters():
            if "classifier" in name:
                param.requires_grad = True

        for m in model_dst.modules():
            if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
                m.eval()    

        return model_dst


if __name__ == "__main__":
    loader= get_datasets()

    train_cfg = BaseTrainCfg(num_experiments=1)
    train_cfg.root_path = os.path.dirname(__file__)
    vgg_cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']

    model_src = VGG(w=1, cfg=vgg_cfg, classes=10)
    load_model(model_src, train_cfg.root_path + "/pt_models/fuse_vgg_cifar_split.pt")

    train_cfg.models = {
        0: {
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
            "epochs" : 40,
            "device": "cuda",
            "mod": ModelModifier(model_src),
            "optimizer": {
                "class": SGD,
                "args": {
                    "lr": 0.05,
                    "momentum": 0.9
                }
            }
        }
    }
    train_cfg.loaders = {
        0: loader
    }
    train_cfg.names = {
        0: "vgg_retrained_head",
    }

    train_from_cfg(train_cfg)