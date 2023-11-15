from net_models.mlp import MLP
from neural_align import NeuralAlign
import eval_tools

import tqdm
import copy
import torch
import argparse
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


def main():
    h          = 128
    layers     = 5
    device     = torch.device("mps")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    FashionMNISTTrainSet = torchvision.datasets.FashionMNIST(
        root='./data', 
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
        root='./data', 
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

    # FashionMnistSet = torchvision.datasets.FashionMNIST(
    #     root='./data', 
    #     train=False,
    #     download=True, 
    #     transform=transform
    # )
    # FashionMnistLoader = torch.utils.data.DataLoader(
    #     FashionMnistSet, 
    #     batch_size=128,
    #     shuffle=True, 
    #     num_workers=8
    # )

    ConcatTrainLoader = torch.utils.data.DataLoader(
        ConcatDataset((FashionMNISTTrainSet, MNISTTrainSet)), 
        batch_size=128,
        shuffle=True, 
        num_workers=8
    )

    model0 = MLP(h, layers).to(device)
    model1 = MLP(h, layers).to(device)
    load_model(model0, "./mlps2/fash_mnist_mlp_e50_l5_h128_v1.pt")
    load_model(model1, "./mlps2/mnist_mlp_e50_l5_h128_v2.pt")

    plain_acc               = list()
    permute_acc             = list()
    permute_and_rescale_acc = list()

    num_experiments = 10
    neural_align_ = NeuralAlign()
    # for i in tqdm.tqdm(range(num_experiments)):
    #     model0_ = copy.deepcopy(model0)
    #     model1_ = copy.deepcopy(model1)

    #     modela = neural_align_.fuse_networks(model0_, model1_, i / 10.0, layers, device=device, loader=ConcatTrainLoader, new_stats=False, permute=False).to(device)

    #     plain_acc.append(eval.evaluate_acc(modela, loader=ConcatTrainLoader, device=device))
    
    # plt.plot(np.linspace(0, 1.0, num_experiments), plain_acc)
    # plt.xlabel("alpha")
    # plt.ylabel("acc")
    # plt.legend(["plain fusion"])
    # plt.savefig("./plots/plain.png")


    for i in tqdm.tqdm(range(num_experiments)):
        model0_ = copy.deepcopy(model0)
        model1_ = copy.deepcopy(model1)

        modela = neural_align_.fuse_networks(model0_, model1_, i / 10.0, layers, device=device, loader=ConcatTrainLoader, new_stats=False, permute=True).to(device)

        permute_acc.append(eval_tools.evaluate_acc(modela, loader=FashionMNISTTrainLoader, device=device))
    
    plt.figure()
    # plt.plot(np.linspace(0, 1.0, num_experiments), plain_acc)
    plt.plot(np.linspace(0, 1.0, num_experiments), permute_acc)
    plt.xlabel("alpha")
    plt.ylabel("acc")
    plt.legend(["plain fusion", "permuted fusion"])
    plt.savefig("./plots/permute.png")


    for i in tqdm.tqdm(range(10)):
        model0_ = copy.deepcopy(model0)
        model1_ = copy.deepcopy(model1)

        modela = neural_align_.fuse_networks(model0_, model1_, i / 10.0, layers, device=device, loader=ConcatTrainLoader, new_stats=True, permute=True).to(device)

        permute_and_rescale_acc.append(eval_tools.evaluate_acc(modela, loader=FashionMNISTTrainLoader, device=device))
    
    plt.figure()
    # plt.plot(np.linspace(0, 1.0, num_experiments), plain_acc)
    plt.plot(np.linspace(0, 1.0, num_experiments), permute_acc)
    plt.plot(np.linspace(0, 1.0, num_experiments), permute_and_rescale_acc)
    plt.xlabel("alpha")
    plt.ylabel("acc")
    plt.legend(["plain fusion", "permuted fusion", "REPAIR fusion"])
    plt.savefig("./plots/permute.png")
   
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', default="cuda")
    parser.add_argument('-m0', '--model0_path', default="cuda")
    parser.add_argument('-m1', '--model1_path', default="cuda")
    args = parser.parse_args()

    main(args.model0_path, args.model1_path, device=args.device)