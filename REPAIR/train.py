import os
from tqdm import tqdm
import torch
import torchvision

import numpy as np
import torchvision.transforms as transforms

from torch.nn import CrossEntropyLoss
from torch.optim import SGD, lr_scheduler

from net_models.mlp import MLP
from REPAIR.eval import evaluate_acc


def save_model(model, i):
    sd = model.state_dict()
    torch.save(sd, 'mlps2/%s.pt' % i)

def load_model(model, i):
    sd = torch.load('mlps2/%s.pt' % i)
    model.load_state_dict(sd)


def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


def train(save_key, layers=5, h=512, train_loader=None, device=None):
    model = MLP(h=h, layers=layers).to(device)
    model.apply(init_weights)

    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    EPOCHS = 20
    ne_iters = len(train_loader)
    # lr_schedule = np.interp(np.arange(1+EPOCHS*ne_iters), [0, 5 * ne_iters, EPOCHS * ne_iters], [0, 1, 0])
    # scheduler = lr_scheduler.LambdaLR(optimizer, lr_schedule.__getitem__)

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
            # scheduler.step()

        print("LOSS: ", loss_acum / total)
    
    save_model(model, save_key)
    print("ACC:", evaluate_acc(model, loader=train_loader, device=device))


def main():
    os.makedirs('./mlps2', exist_ok=True)
    device = torch.device("cuda")
    MNIST_MEAN = [0.1307]
    MNIST_STD = [0.3081]

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.Normalize(np.array(MNIST_MEAN), np.array(MNIST_STD))
        ]
    )

    FashMnistTrainSet = torchvision.datasets.FashionMNIST(
        root='./data', 
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
        root='./data', 
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

    #FASHION MNIST
    #1: 0.92535
    #2: 0.92591
    h = 128
    layers = 5
    train('mnist_mlp_e50_l%d_h%d_v1_cuda' % (layers, h), layers=layers, h=h, train_loader=mnistTrainLoader, device=device)  
    train('fash_mnist_mlp_e50_l%d_h%d_v2_cuda' % (layers, h), layers=layers, h=h, train_loader=FashMnistTrainLoader, device=device)


if __name__ == "__main__":
    main()