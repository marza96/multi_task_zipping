import os
import torch
import argparse
import torchvision

import numpy as np
import torchvision.transforms as transforms
from torchvision.transforms.functional import rotate


from tqdm import tqdm
# from torch.nn import CrossEntropyLoss
# from torch.optim import SGD, lr_scheduler

# from net_models.mlp import MLP
from .eval_tools import evaluate_acc, evaluate_acc_single_head


def save_model(model, i):
    sd = model.state_dict()
    torch.save(sd, i)

def load_model(model, i):
    sd = torch.load(i)
    model.load_state_dict(sd)


def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


def transp(x):
    return x.permute(0, 2, 1)


# def train(save_key, layers=5, h=512, train_loader=None, device=None, idx=None):
#     model = MLP(h=h, layers=layers).to(device)
#     model.apply(init_weights)

#     optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    
#     EPOCHS = 20

#     loss_fn = CrossEntropyLoss()

#     for _ in tqdm(range(EPOCHS)):
#         model.train()
#         loss_acum = 0.0
#         total = 0
#         for i, (inputs, labels) in enumerate(train_loader):
#             optimizer.zero_grad(set_to_none=True)

#             outputs = model(inputs.to(device))
#             loss    = loss_fn(outputs, labels.to(device))

#             loss.backward()

#             loss_acum += loss.mean()
#             total += 1
#             optimizer.step()

#         print("LOSS: ", loss_acum / total)
    
    

#     save_model(model, save_key)
#     print("ACC:", evaluate_acc_single_head(model, loader=train_loader, device=device))


def train_loop(*, model, optimizer, loss_fn, epochs, train_loader, device="cuda"):
    model.apply(init_weights)

    for _ in tqdm(range(epochs)):
        model.train()
        loss_acum = 0.0
        total = 0
        for _, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)

            outputs = model(inputs.to(device))
            loss    = loss_fn(outputs, labels.to(device))

            loss.backward()

            loss_acum += loss.mean()
            total += 1
            optimizer.step()

        print("LOSS: ", loss_acum / total)
    
    print("TRAIN ACC:", evaluate_acc_single_head(model, loader=train_loader, device=device))

    return model


def train_from_cfg(train_cfg):
    for i in range(train_cfg.num_experiments):
        model_cls      = train_cfg.models[i]["model"]
        model_args     = train_cfg.models[i]["args"]
        optimizer_args = train_cfg.configs[i]["optimizer"]["args"]
        optimizer_cls  = train_cfg.configs[i]["optimizer"]["class"]
        epochs         = train_cfg.configs[i]["epochs"]
        device         = train_cfg.configs[i]["device"]
        loss_fn        = train_cfg.configs[i]["loss_fn"]
        train_loader   = train_cfg.loaders[i]
        exp_name       = train_cfg.names[i]
        root_path      = train_cfg.root_path

        print("DEV", device)
        model          = model_cls(**model_args).to(device)
        optimizer      = optimizer_cls(model.parameters(), **optimizer_args)

        trained_model = train_loop(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            epochs=epochs,
            train_loader=train_loader,
            device=device
        )

        os.makedirs("%s/pt_models/" %(root_path), exist_ok=True)
        save_model(trained_model, "%s/pt_models/%s.pt" %(root_path, exp_name))

def main(dataset0, dataset1, device="cuda"):
    device = torch.device(device)
    path   = os.path.dirname(__file__)
    os.makedirs(path + '/pt_models', exist_ok=True)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

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

    mnistTrainLoader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(mnistTrainSet, second_half),
        batch_size=128,
        num_workers=8)


    h = 512
    layers = 6

    loader0 = None
    loader1 = None
    prefix0 = None
    prefix1 = None

    if dataset0 == "MNIST":
        loader0 = mnistTrainLoader
        prefix0 = "mnist_"
    elif dataset0 == "FashionMNIST":
        loader0 = RMnistTrainLoader
        prefix0 = "fash_mnist_"

    if dataset1 == "MNIST":
        loader1 = mnistTrainLoader
        prefix1 = "mnist_"
    elif dataset1 == "FashionMNIST":
        loader1 = RMnistTrainLoader
        prefix1 = "fash_mnist_"

    print(loader0, prefix0)
    train(path + '/pt_models/%smlp_e50_l%d_h%d_v1_%s.pt' % (prefix0, layers, h, device), layers=layers, h=h, train_loader=loader0, device=device, idx=0)  
    train(path + '/pt_models/%smlp_e50_l%d_h%d_v2_%s.pt' % (prefix1, layers, h, device), layers=layers, h=h, train_loader=loader1, device=device, idx=1)  
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', default="cuda")
    parser.add_argument('-d0', '--dataset0', default="cuda")
    parser.add_argument('-d1', '--dataset1', default="cuda")
    args = parser.parse_args()

    main(args.dataset0, args.dataset1, device=args.device)