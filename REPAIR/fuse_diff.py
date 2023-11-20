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

def transp(x):
    return x.permute(0, 2, 1)


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
        os.makedirs(root_path + '/plots/diff', exist_ok=True)

        neural_align_ = NeuralAlignDiff(model_cls, loader0, loader1, loaderc)
        permute_and_rescale_acc   = list()
        permute_acc   = list()
        plain_acc     = list()

        for i in tqdm.tqdm(range(alpha_split)):
            model0_ = copy.deepcopy(model0)
            model1_ = copy.deepcopy(model1)

            modela = neural_align_.fuse_networks(model_args, model0_, model1_, i / 10.0, device=device, new_stats=False, permute=False).to(device)

            plain_acc.append(evaluate_acc_single_head(modela, loader=loaderc, device=device))
            if i == 5:
                print("plain acc: ", plain_acc[-1])

        plt.figure()
        plt.plot(np.linspace(0, 1.0, alpha_split), plain_acc)
        plt.xlabel("alpha")
        plt.ylabel("acc")
        plt.legend(["plain fusion"])
        plt.savefig("%s/plots/diff/plain.png" % root_path)

        for i in tqdm.tqdm(range(alpha_split)):
            model0_ = copy.deepcopy(model0)
            model1_ = copy.deepcopy(model1)

            modela = neural_align_.fuse_networks(model_args, model0_, model1_, i / 10.0, device=device, new_stats=False, permute=True).to(device)

            permute_acc.append(evaluate_acc_single_head(modela, loader=loaderc, device=device))

            if i == 5:
                print("permute acc: ", permute_acc[-1])

        plt.plot(np.linspace(0, 1.0, alpha_split), permute_acc)
        plt.xlabel("alpha")
        plt.ylabel("acc")
        plt.legend(["plain fusion", "permuted fusion"])
        plt.savefig("%s/plots/diff/permute.png" % root_path)

        for i in tqdm.tqdm(range(alpha_split)):
            model0_ = copy.deepcopy(model0)
            model1_ = copy.deepcopy(model1)

            modela = neural_align_.fuse_networks(model_args, model0_, model1_, i / 10.0, device=device, new_stats=True, permute=True).to(device)

            permute_and_rescale_acc.append(evaluate_acc_single_head(modela, loader=loaderc, device=device))

            if i == 5:
                print("permute_and_scale acc: ", permute_and_rescale_acc[-1])

        plt.plot(np.linspace(0, 1.0, alpha_split), permute_and_rescale_acc)
        plt.xlabel("alpha")
        plt.ylabel("acc")
        plt.legend(["plain fusion", "permuted fusion", "REPAIR fusion"])
        plt.savefig("%s/plots/diff/repair.png" % root_path)


def main(model0_path, model1_path, device="cuda"):
    h          = 512
    layers     = 5
    device     = torch.device(device)
    path       = os.path.dirname(__file__)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    # Skype handle: aeniha
    # FashionMNISTTrainSet = torchvision.datasets.FashionMNIST(
    #     root=path + '/data', 
    #     train=True,
    #     download=True, 
    #     transform=transform,
    #     target_transform=util.OffsetLabel(10)
    # )
    # FashionMNISTTrainLoader = torch.utils.data.DataLoader(
    #     FashionMNISTTrainSet, 
    #     batch_size=128,
    #     shuffle=True, 
    #     num_workers=8
    # )
    # MNISTTrainSet = torchvision.datasets.MNIST(
    #     root=path + '/data', 
    #     train=False,
    #     download=True, 
    #     transform=transform
    # )
    # MNISTTrainLoader = torch.utils.data.DataLoader(
    #     MNISTTrainSet, 
    #     batch_size=128,
    #     shuffle=False, 
    #     num_workers=8
    # )

    # RMnistTrainSet = torchvision.datasets.MNIST(
    #     root=path + '/data', 
    #     train=False,
    #     download=True, 
    #     transform=transforms.Compose([
    #         transforms.Lambda(lambda x: rotate(x, 45, fill=(0,),
    #             interpolation=transforms.InterpolationMode.BILINEAR)),
    #         transforms.ToTensor()])
    # )
    # RMnistTrainLoader = torch.utils.data.DataLoader(
    #     RMnistTrainSet, 
    #     batch_size=128,
    #     shuffle=False, 
    #     num_workers=8
    # )


    mnistTrainSet = torchvision.datasets.MNIST(
        root=path + '/data', 
        train=True,
        download=True, 
        transform=transform
    )

    first_half = [0, 1, 2, 3, 4]
    second_half = [5, 6, 7, 8, 9]
    first_half = [idx for idx, target in enumerate(mnistTrainSet.targets) if target in first_half]
    second_half = [idx for idx, target in enumerate(mnistTrainSet.targets) if target in second_half]
    
    RMnistTrainLoader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(mnistTrainSet, first_half),
        batch_size=128,
        num_workers=8)

    MNISTTrainLoader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(mnistTrainSet, second_half),
        batch_size=128,
        num_workers=8)

    ConcatTrainLoader = torch.utils.data.DataLoader(
        ConcatDataset((torch.utils.data.Subset(mnistTrainSet, first_half), torch.utils.data.Subset(mnistTrainSet, second_half))), 
        batch_size=128,
        shuffle=False, 
        num_workers=8
    )


    model0 = MLP(channels=128, layers=5, classes=10).to(device)
    model1 = MLP(channels=128, layers=5, classes=10).to(device)


    # load_model(model0, path + "/pt_models/" + model0_path)
    # load_model(model1, path + "/pt_models/" + model1_path)

    load_model(model0, "/Users/harissikic/Downloads/multi_task_zipping 3/pt_models/mlp_first.pt")
    load_model(model1, "/Users/harissikic/Downloads/multi_task_zipping 3/pt_models/mlp_second.pt")

    print("ACC", evaluate_acc_single_head(model1, loader=MNISTTrainLoader, device=device))

    plain_acc               = list()
    permute_acc             = list()
    permute_and_rescale_acc = list()

    num_experiments = 10
    neural_align_ = NeuralAlignDiff(RMnistTrainLoader, MNISTTrainLoader, ConcatTrainLoader)

    # model0_ = copy.deepcopy(model0)
    # model1_ = copy.deepcopy(model1)
    # modela_ = neural_align_.fuse_networks(model_args, model0_, model1_, 0.5, layeew_stats=False, permute=True).to(device)

    # model0_ = wrap_layers(model0_, rescale=False)
    # modela_ = wrap_layers(modela_, rescale=False)
    # stats0 = estimate_stats(model0_, RMnistTrainLoader, device=device)
    # statsa = estimate_stats(modela_, RMnistTrainLoader, device=device)

    # print("DBG per", torch.mean(statsa[3][1] / stats0[3][1]))

    # model0_ = copy.deepcopy(model0)
    # model1_ = copy.deepcopy(model1)
    # modela_ = neural_align_.fuse_networks(model_args, model0_, model1_, 0.5, layeew_stats=True, permute=True).to(device)

    # stats0 = estimate_stats(model0_, RMnistTrainLoader, device=device)
    # statsa = estimate_stats(modela_, RMnistTrainLoader, device=device)

    # print("DBG rep", torch.mean(statsa[3][1] / stats0[3][1]))

    for i in tqdm.tqdm(range(num_experiments)):
        model0_ = copy.deepcopy(model0)
        model1_ = copy.deepcopy(model1)

        modela = neural_align_.fuse_networks(model_args, model0_, model1_, i / 10.0,ce, new_stats=False, permute=False).to(device)

        plain_acc.append(evaluate_acc_single_head(modela, loader=ConcatTrainLoader, device=device))
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

        modela = neural_align_.fuse_networks(model_args, model0_, model1_, i / 10.0,ce, new_stats=False, permute=True).to(device)

        permute_acc.append(evaluate_acc_single_head(modela, loader=ConcatTrainLoader, device=device))

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

        modela = neural_align_.fuse_networks(model_args, model0_, model1_, i / 10.0,ce, new_stats=True, permute=True).to(device)

        permute_and_rescale_acc.append(evaluate_acc_single_head(modela, loader=ConcatTrainLoader, device=device))

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
    parser.add_argument('-d', '--device', default="mps")
    parser.add_argument('-m0', '--model0_path', default="cuda")
    parser.add_argument('-m1', '--model1_path', default="cuda")
    args = parser.parse_args()

    main(args.model0_path, args.model1_path, device=args.device)