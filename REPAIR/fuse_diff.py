from net_models.mlp import MLP
import eval_tools
from neural_align_diff import NeuralAlignDiff

import argparse

import os
import tqdm
import copy
import torch
import torchvision

import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from torch.utils.data import ConcatDataset, DataLoader


def save_model(model, i):
    sd = model.state_dict()
    torch.save(sd, i)

def load_model(model, i):
    sd = torch.load(i)
    model.load_state_dict(sd)


def main(model0_path, model1_path, device="cuda"):
    h          = 128
    layers     = 5
    device     = torch.device(device)
    path       = os.path.dirname(__file__)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    FashionMNISTTrainSet = torchvision.datasets.FashionMNIST(
        root=path + '/data', 
        train=True,
        download=True, 
        transform=transform
    )
    FashionMNISTTrainLoader = torch.utils.data.DataLoader(
        FashionMNISTTrainSet, 
        batch_size=128,
        shuffle=True, 
        num_workers=8
    )
    MNISTTrainSet = torchvision.datasets.MNIST(
        root=path + '/data', 
        train=True,
        download=True, 
        transform=transform
    )
    MNISTTrainLoader = torch.utils.data.DataLoader(
        MNISTTrainSet, 
        batch_size=128,
        shuffle=True, 
        num_workers=8
    )

    ConcatTrainLoader = torch.utils.data.DataLoader(
        ConcatDataset((FashionMNISTTrainSet, MNISTTrainSet)), 
        batch_size=128,
        shuffle=True, 
        num_workers=8
    )

    model0 = MLP(h, layers).to(device)
    model1 = MLP(h, layers).to(device)

    load_model(model0, path + "/pt_models/" + model0_path)
    load_model(model1, path + "/pt_models/" + model1_path)

    plain_acc               = list()
    permute_acc             = list()
    permute_and_rescale_acc = list()

    num_experiments = 10
    neural_align_ = NeuralAlignDiff(FashionMNISTTrainLoader, MNISTTrainLoader, ConcatTrainLoader)
    for i in tqdm.tqdm(range(num_experiments)):
        model0_ = copy.deepcopy(model0)
        model1_ = copy.deepcopy(model1)

        modela = neural_align_.fuse_networks(model0_, model1_, i / 10.0, layers, device=device, new_stats=False, permute=False).to(device)

        plain_acc.append(eval_tools.evaluate_acc(modela, loader=ConcatTrainLoader, device=device))
    
    plt.figure()
    plt.plot(np.linspace(0, 1.0, num_experiments), plain_acc)
    plt.xlabel("alpha")
    plt.ylabel("acc")
    plt.legend(["plain fusion"])
    plt.savefig(path + "/plots/diff/plain.png")


    for i in tqdm.tqdm(range(num_experiments)):
        model0_ = copy.deepcopy(model0)
        model1_ = copy.deepcopy(model1)

        modela = neural_align_.fuse_networks(model0_, model1_, i / 10.0, layers, device=device, new_stats=False, permute=True).to(device)

        permute_acc.append(eval_tools.evaluate_acc(modela, loader=ConcatTrainLoader, device=device))
    
    # plt.plot(np.linspace(0, 1.0, num_experiments), plain_acc)
    plt.plot(np.linspace(0, 1.0, num_experiments), permute_acc)
    plt.xlabel("alpha")
    plt.ylabel("acc")
    plt.legend(["plain fusion", "permuted fusion"])
    plt.savefig(path + "/plots/diff/permute.png")


    for i in tqdm.tqdm(range(10)):
        model0_ = copy.deepcopy(model0)
        model1_ = copy.deepcopy(model1)

        modela = neural_align_.fuse_networks(model0_, model1_, i / 10.0, layers, device=device, new_stats=True, permute=True).to(device)

        permute_and_rescale_acc.append(eval_tools.evaluate_acc(modela, loader=ConcatTrainLoader, device=device))
    
    # plt.plot(np.linspace(0, 1.0, num_experiments), plain_acc)
    plt.plot(np.linspace(0, 1.0, num_experiments), permute_and_rescale_acc)
    plt.xlabel("alpha")
    plt.ylabel("acc")
    plt.legend(["plain fusion", "permuted fusion", "REPAIR fusion"])
    plt.savefig(path + "/plots/diff/repair.png")
   

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', default="cuda")
    parser.add_argument('-m0', '--model0_path', default="cuda")
    parser.add_argument('-m1', '--model1_path', default="cuda")
    args = parser.parse_args()

    main(args.model0_path, args.model1_path, device=args.device)