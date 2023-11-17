from net_models.mlp import MLP, LayerWrapper
import eval_tools
from neural_align_diff import NeuralAlignDiff

import argparse

import os
import tqdm
import util
import copy
import torch
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

def unwrap_layers(model, rescale):
    unwrapped_model = model

    unwrapped_model.fc1 = unwrapped_model.fc1.layer
    
    for i in range(len(unwrapped_model.layers)):
        layer = unwrapped_model.layers[i]

        if isinstance(layer, nn.Linear):
            unwrapped_model.layers[i] = unwrapped_model.layers[i].layer

    return unwrapped_model


def wrap_layers(model, rescale):
    wrapped_model = model

    wrapped_model.fc1 = LayerWrapper(wrapped_model.fc1, rescale=rescale)

    for i in range(len(wrapped_model.layers)):
        layer = wrapped_model.layers[i]

        if isinstance(layer, nn.Linear):
            wrapped_model.layers[i] = LayerWrapper(wrapped_model.layers[i], rescale=rescale)

    return wrapped_model


def estimate_stats(model, loader, device=None, rescale=False):
    statistics = list()
    model = model.to(device)
    
    model.train()
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.momentum = None
            m.reset_running_stats()

    with tqdm.tqdm(torch.no_grad()):
        for inputs, labels in loader:
            o2 = model(inputs.to(device))

    model.eval()

    stats_ = model.fc1.get_stats()
    statistics.append(stats_)


    for i in tqdm.tqdm(range(len(model.layers))):
        if not isinstance(model.layers[i], LayerWrapper):
            continue

        stats_ = model.layers[i].get_stats()

        statistics.append(stats_)

    return statistics


def main(model0_path, model1_path, device="cuda"):
    h          = 512
    layers     = 3
    device     = torch.device(device)
    path       = os.path.dirname(__file__)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    # Skype handle: aeniha
    FashionMNISTTrainSet = torchvision.datasets.FashionMNIST(
        root=path + '/data', 
        train=False,
        download=True, 
        transform=transform,
        target_transform=util.OffsetLabel(10)
    )
    FashionMNISTTrainLoader = torch.utils.data.DataLoader(
        FashionMNISTTrainSet, 
        batch_size=128,
        shuffle=False, 
        num_workers=8
    )
    MNISTTrainSet = torchvision.datasets.MNIST(
        root=path + '/data', 
        train=False,
        download=True, 
        transform=transform
    )
    MNISTTrainLoader = torch.utils.data.DataLoader(
        MNISTTrainSet, 
        batch_size=128,
        shuffle=False, 
        num_workers=8
    )

    ConcatTrainLoader = torch.utils.data.DataLoader(
        ConcatDataset((FashionMNISTTrainSet, MNISTTrainSet)), 
        batch_size=128,
        shuffle=False, 
        num_workers=8
    )


    model0 = MLP(h, layers).to(device)
    model1 = MLP(h, layers).to(device)

    load_model(model0, path + "/pt_models/" + model0_path)
    load_model(model1, path + "/pt_models/" + model1_path)

    print("ACC", eval_tools.evaluate_acc(model1, loader=ConcatTrainLoader, device=device))

    plain_acc               = list()
    permute_acc             = list()
    permute_and_rescale_acc = list()

    num_experiments = 10
    neural_align_ = NeuralAlignDiff(FashionMNISTTrainLoader, MNISTTrainLoader, ConcatTrainLoader)

    # model0_ = copy.deepcopy(model0)
    # model1_ = copy.deepcopy(model1)
    # modela_ = neural_align_.fuse_networks(model0_, model1_, 0.5, layers, device=device, new_stats=False, permute=True).to(device)

    # model0_ = wrap_layers(model0_, rescale=False)
    # modela_ = wrap_layers(modela_, rescale=False)
    # stats0 = estimate_stats(model0_, FashionMNISTTrainLoader, device=device)
    # statsa = estimate_stats(modela_, FashionMNISTTrainLoader, device=device)

    # print("DBG per", torch.mean(statsa[3][1] / stats0[3][1]))

    # model0_ = copy.deepcopy(model0)
    # model1_ = copy.deepcopy(model1)
    # modela_ = neural_align_.fuse_networks(model0_, model1_, 0.5, layers, device=device, new_stats=True, permute=True).to(device)

    # stats0 = estimate_stats(model0_, FashionMNISTTrainLoader, device=device)
    # statsa = estimate_stats(modela_, FashionMNISTTrainLoader, device=device)

    # print("DBG rep", torch.mean(statsa[3][1] / stats0[3][1]))

    for i in tqdm.tqdm(range(num_experiments)):
        model0_ = copy.deepcopy(model0)
        model1_ = copy.deepcopy(model1)

        modela = neural_align_.fuse_networks(model0_, model1_, i / 10.0, layers, device=device, new_stats=False, permute=False).to(device)

        plain_acc.append(eval_tools.evaluate_acc(modela, loader=ConcatTrainLoader, device=device))
        if i == 5:
            print("plain acc: ", plain_acc[-1])

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

        if i == 5:
            print("permute acc: ", permute_acc[-1]) 

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

        if i == 5:
            print("repair acc: ", permute_and_rescale_acc[-1]) 

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