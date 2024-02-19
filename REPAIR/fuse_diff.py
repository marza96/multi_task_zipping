from .eval_tools import evaluate_acc_single_head
from .neural_align_diff import NeuralAlignDiff

from torchvision.transforms.functional import rotate

import os
import tqdm
import copy
import torch
import wandb

import numpy as np
import matplotlib.pyplot as plt


def save_model(model, i):
    sd = model.state_dict()
    torch.save(sd, i)


def load_model(model, i):
    sd = torch.load(i, map_location=torch.device('cpu'))
    model.load_state_dict(sd)


def plot_stuff(x, y, x_label, y_label, legend, path):
    plt.plot(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(legend)
    plt.savefig(path)


def fuse_from_cfg(train_cfg, debug=True):
    for i in range(0, train_cfg.num_experiments):
        model_cls      = train_cfg.models[i]["model"]
        model_args     = train_cfg.models[i]["args"]
        loader0        = train_cfg.loaders[i]["loader0"]
        loader1        = train_cfg.loaders[i]["loader1"]
        loaderc        = train_cfg.loaders[i]["loaderc"]
        exp_name       = train_cfg.names[i]["experiment_name"]
        model0_name    = train_cfg.names[i]["model0_name"]
        model1_name    = train_cfg.names[i]["model1_name"]
        device         = train_cfg.configs[i]["device"]
        match_method   = train_cfg.configs[i]["match_method"]
        model_mod      = train_cfg.configs[i]["model_mod"]
        root_path      = train_cfg.root_path
        alpha_split    = train_cfg.alpha_split
        proj_name      = train_cfg.proj_name

        model0          = model_cls(**model_args)
        model1          = model_cls(**model_args)

        proj_name = train_cfg.proj_name

        desc = {"experiment": exp_name}

        wandb.init(
            project=proj_name,
            config=desc,
            name=exp_name
        )

        if model_mod is not None:
            model0 = model_mod(model0)
            model1 = model_mod(model1)
            
        load_model(model0, "%s/pt_models/%s.pt" %(root_path, model0_name))
        load_model(model1, "%s/pt_models/%s.pt" %(root_path, model1_name))

        os.makedirs(root_path + '/plots', exist_ok=True)
        os.makedirs(root_path + '/plots/%s' % exp_name, exist_ok=True)

        neural_align_ = NeuralAlignDiff(model_cls, match_method, loader0, loader1, loaderc)
        
        # modela  = neural_align_.fuse_networks(model_args, model0, model1, 0.5, device=device, new_stats=True, permute=True, model_mod=model_mod).to(device)
        # acc     = evaluate_acc_single_head(modela.to(device), loader=loaderc, device=device)
        # print("ITER %d Fused model accuracy: %f" %(i, acc))
        # return
        
        for i in tqdm.tqdm(range(alpha_split)):
            model0_ = copy.deepcopy(model0)
            model1_ = copy.deepcopy(model1)

            modela = neural_align_.fuse_networks(model_args, model0_, model1_, i / alpha_split, device=device, new_stats=True, permute=True).to(device)
            
            acc = evaluate_acc_single_head(modela, loader=loaderc, device=device)

            wandb.log({"REPAIR acc": acc})

        for i in tqdm.tqdm(range(alpha_split)):
            model0_ = copy.deepcopy(model0)
            model1_ = copy.deepcopy(model1)

            modela = neural_align_.fuse_networks(model_args, model0_, model1_, i / alpha_split, device=device, new_stats=False, permute=True).to(device)

            acc = evaluate_acc_single_head(modela, loader=loaderc, device=device)

            wandb.log({"permuted acc": acc})

        wandb.finish()
