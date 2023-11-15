import os
import torch
import argparse
import torchvision

import numpy as np
import torchvision.transforms as transforms

from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, lr_scheduler

from net_models.mlp import MLP
from eval_tools import evaluate_acc


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


def train(save_key, layers=5, h=512, train_loader=None, device=None):
    model = MLP(h=h, layers=layers).to(device)
    model.apply(init_weights)

    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    EPOCHS = 1

    loss_fn = CrossEntropyLoss()

    for _ in tqdm(range(EPOCHS)):
        model.train()
        loss_acum = 0.0
        total = 0
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)

            outputs = model(inputs.to(device))
            loss    = loss_fn(outputs, labels.to(device))

            loss.backward()

            loss_acum += loss.mean()
            total += 1
            optimizer.step()

        print("LOSS: ", loss_acum / total)
    
    save_model(model, save_key)
    print("ACC:", evaluate_acc(model, loader=train_loader, device=device))


def main(dataset0, dataset1, device="cuda"):
    device = torch.device(device)
    path   = os.path.dirname(__file__)
    os.makedirs(path + '/pt_models', exist_ok=True)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    FashMnistTrainSet = torchvision.datasets.FashionMNIST(
        root=path + '/data', 
        train=True,
        download=True, 
        transform=transform
    )
    FashMnistTrainLoader = torch.utils.data.DataLoader(
        FashMnistTrainSet, 
        batch_size=128,
        shuffle=True, 
        num_workers=8
    )

    mnistTrainSet = torchvision.datasets.MNIST(
        root=path + '/data', 
        train=True,
        download=True, 
        transform=transform
    )
    mnistTrainLoader = torch.utils.data.DataLoader(
        mnistTrainSet, 
        batch_size=128,
        shuffle=True, 
        num_workers=8
    )

    h = 128
    layers = 5

    loader0 = None
    loader1 = None
    prefix0 = None
    prefix1 = None

    if dataset0 == "MNIST":
        loader0 = mnistTrainLoader
        prefix0 = "mnist_"
    elif dataset0 == "FashionMNIST":
        loader0 = FashMnistTrainLoader
        prefix0 = "fash_mnist_"

    if dataset1 == "MNIST":
        loader1 = mnistTrainLoader
        prefix1 = "mnist_"
    elif dataset1 == "FashionMNIST":
        loader1 = FashMnistTrainLoader
        prefix1 = "fash_mnist_"

    train(path + '/pt_models/%smlp_e50_l%d_h%d_v1_%s.pt' % (prefix0, layers, h, device), layers=layers, h=h, train_loader=loader0, device=device)  
    train(path + '/pt_models/%smlp_e50_l%d_h%d_v2_%s.pt' % (prefix1, layers, h, device), layers=layers, h=h, train_loader=loader1, device=device)  
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', default="cuda")
    parser.add_argument('-d0', '--dataset0', default="cuda")
    parser.add_argument('-d1', '--dataset1', default="cuda")
    args = parser.parse_args()

    main(args.dataset0, args.dataset1, device=args.device)