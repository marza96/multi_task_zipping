from .net_models.mlp import MLP, LayerWrapper
from .eval_tools import evaluate_acc_single_head
from .neural_align_diff import NeuralAlignDiff

from torchvision.transforms.functional import rotate

import os
import tqdm
import copy
import torch
import argparse
import torchvision

import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from torch.utils.data import ConcatDataset, DataLoader


def save_model(model, i):
    sd = model.state_dict()
    torch.save(sd, i)


def load_model(model, i):
    sd = torch.load(i)
    model.load_state_dict(sd)


def plot_stuff(x, y, x_label, y_label, legend, path):
    plt.plot(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(legend)
    plt.savefig(path)


def fuse_from_cfg(train_cfg):
    for i in range(train_cfg.num_experiments):
        model_cls      = train_cfg.models[i]["model"]
        model_args     = train_cfg.models[i]["args"]
        loss_fn        = train_cfg.configs[i]["loss_fn"]
        loader0        = train_cfg.loaders[i]["loader0"]
        loader1        = train_cfg.loaders[i]["loader1"]
        loaderc        = train_cfg.loaders[i]["loaderc"]
        exp_name       = train_cfg.names[i]["experiment_name"]
        model0_name    = train_cfg.names[i]["model0_name"]
        model1_name    = train_cfg.names[i]["model1_name"]
        device         = train_cfg.configs[i]["device"]
        root_path      = train_cfg.root_path
        alpha_split    = train_cfg.alpha_split

        model0          = model_cls(**model_args)
        model1          = model_cls(**model_args)

        load_model(model0, "%s/pt_models/%s.pt" %(root_path, model0_name))
        load_model(model1, "%s/pt_models/%s.pt" %(root_path, model1_name))

        os.makedirs(root_path + '/plots', exist_ok=True)
        os.makedirs(root_path + '/plots/%s' % exp_name, exist_ok=True)

        neural_align_ = NeuralAlignDiff(model_cls, loader0, loader1, loaderc)
        permute_and_rescale_acc   = list()
        permute_acc   = list()
        plain_acc     = list()

        for i in tqdm.tqdm(range(alpha_split)):
            model0_ = copy.deepcopy(model0)
            model1_ = copy.deepcopy(model1)

            modela = neural_align_.fuse_networks(model_args, model0_, model1_, i / 10.0, device=device, new_stats=False, permute=False).to(device)

            plain_acc.append(evaluate_acc_single_head(modela, loader=loaderc, device=device))
            if i == alpha_split / 2:
                print("plain acc: ", plain_acc[-1])

        plt.figure()

        plot_stuff(
            np.linspace(0, 1.0, alpha_split), 
            plain_acc, 
            "alpha", 
            "acc", 
            ["plain fusion"], 
            root_path + '/plots/%s/plain.png' % exp_name
        )

        for i in tqdm.tqdm(range(alpha_split)):
            model0_ = copy.deepcopy(model0)
            model1_ = copy.deepcopy(model1)

            modela = neural_align_.fuse_networks(model_args, model0_, model1_, i / 10.0, device=device, new_stats=False, permute=True).to(device)

            permute_acc.append(evaluate_acc_single_head(modela, loader=loaderc, device=device))

            if i == alpha_split / 2:
                print("permute acc: ", permute_acc[-1])

        plot_stuff(
            np.linspace(0, 1.0, alpha_split), 
            permute_acc,
            "alpha",
            "acc",
            ["plain fusion", "permuted fusion"],
            root_path + '/plots/%s/permute.png' % exp_name
        )

        for i in tqdm.tqdm(range(alpha_split)):
            model0_ = copy.deepcopy(model0)
            model1_ = copy.deepcopy(model1)

            modela = neural_align_.fuse_networks(model_args, model0_, model1_, i / 10.0, device=device, new_stats=True, permute=True).to(device)

            permute_and_rescale_acc.append(evaluate_acc_single_head(modela, loader=loaderc, device=device))

            if i == alpha_split / 2:
                print("permute_and_scale acc: ", permute_and_rescale_acc[-1])

        plot_stuff(
            np.linspace(0, 1.0, alpha_split), 
            permute_and_rescale_acc,
            "alpha",
            "acc",
            ["plain fusion", "permuted fusion", "REPAIR fusion"],
            root_path + '/plots/%s/repair.png' % exp_name
        )
