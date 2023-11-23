import os
import torch
import argparse
import torchvision

import numpy as np
import torchvision.transforms as transforms
from torchvision.transforms.functional import rotate


from tqdm import tqdm
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
        model_mod      = train_cfg.configs[i]["model_mod"]
        train_loader   = train_cfg.loaders[i]
        exp_name       = train_cfg.names[i]
        root_path      = train_cfg.root_path
        model          = model_cls(**model_args).to(device)
        optimizer      = optimizer_cls(model.parameters(), **optimizer_args)

        if model_mod is not None:
            model = model_mod(model)

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
