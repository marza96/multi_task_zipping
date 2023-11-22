import torch
import scipy
import copy
import math

from tqdm import tqdm

from .net_models.mlp import LayerWrapper, LayerWrapper2D, CNN

import numpy as np
import torch.nn as nn

import matplotlib.pyplot as plt

class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)
    

class StraightThroughEstimator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
            x = STEFunction.apply(x)
            return x
    

class NeuralAlignDiff:
    def __init__(self, model_cls, loader0, loader1, loaderc) -> None:
        self.permutations   = list()
        self.statistics     = list()
        self.layer_indices  = list()
        self.perms_calc     = False
        self.stats_calc     = False
        self.layers_indexed = False

        self.loader0   = loader0
        self.loader1   = loader1
        self.loaderc   = loaderc
        self.model_cls = model_cls

        self.dbg0 = None

    def run_act_similarity(self, net0, net1, epochs=1, loader=None, device=None):
        n = epochs * len(loader)
        F = None

        with torch.no_grad():
            net0.eval()
            net1.eval()

            for _ in range(epochs):
                for i, (images, _) in enumerate(tqdm(loader)):
                    img_t = images.float().to(device)
                    out0 = net0(img_t)
                    out0 = out0.reshape(out0.shape[0], out0.shape[1], -1).permute(0, 2, 1)
                    out0 = out0.reshape(-1, out0.shape[2]).float()

                    out1 = net1(img_t)
                    out1 = out1.reshape(out1.shape[0], out1.shape[1], -1).permute(0, 2, 1)
                    out1 = out1.reshape(-1, out1.shape[2]).float()

                    prod = out0.T @ out1

                    if i == 0:
                        F = torch.zeros(prod.shape).to(device)
                    
                    F += prod

        return F

    def run_corr_matrix(self, net0, net1, epochs=1, loader=None, device=None):
        n = epochs * len(loader)
        mean0 = mean1 = std0 = std1 = outer = None
        with torch.no_grad():
            net0.eval()
            net1.eval()

            for _ in range(epochs):
                for i, (images, _) in enumerate(tqdm(loader)):
                    img_t = images.float().to(device)
                    out0 = net0(img_t)
                    out0 = out0.reshape(out0.shape[0], out0.shape[1], -1).permute(0, 2, 1)
                    out0 = out0.reshape(-1, out0.shape[2]).float()

                    out1 = net1(img_t)
                    out1 = out1.reshape(out1.shape[0], out1.shape[1], -1).permute(0, 2, 1)
                    out1 = out1.reshape(-1, out1.shape[2]).float()

                    mean0_b = out0.mean(dim=0)
                    mean1_b = out1.mean(dim=0)
                    std0_b = out0.std(dim=0)
                    std1_b = out1.std(dim=0)
                    outer_b = (out0.T @ out1) / out0.shape[0]

                    if i == 0:
                        mean0 = torch.zeros_like(mean0_b)
                        mean1 = torch.zeros_like(mean1_b)
                        std0 = torch.zeros_like(std0_b)
                        std1 = torch.zeros_like(std1_b)
                        outer = torch.zeros_like(outer_b)

                    mean0 += mean0_b / n
                    mean1 += mean1_b / n
                    std0 += std0_b / n
                    std1 += std1_b / n
                    outer += outer_b / n

        cov = outer - torch.outer(mean0, mean1)
        corr = cov / (torch.outer(std0, std1) + 1e-4)

        return corr

    def solve_lap(self, corr_mtx):
        corr_mtx_a = corr_mtx.cpu().numpy()
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(corr_mtx_a, maximize=True)

        assert (row_ind == np.arange(len(corr_mtx_a))).all()

        perm_map = torch.tensor(col_ind).long()

        return perm_map
    
    def perm_to_permmat(self, permutation):
        perm_mat = torch.zeros((len(permutation), len(permutation)))
        perm_mat[torch.arange(len(permutation)), permutation] = 1

        return perm_mat

    def permmat_to_perm(self, permmat):
        perm = torch.Tensor(permmat.shape[0])
        perm = torch.where(permmat == 1)[1]

        return perm

    def get_layer_perm(self, net0, net1, epochs=1, loader=None, device=None):
        corr_mtx = self.run_corr_matrix(
            net0, 
            net1, 
            epochs=1, 
            loader=loader, 
            device=device
        )

        return self.solve_lap(corr_mtx), corr_mtx
    
    def perm_coord_descent(self, net0, net1, epochs=300, device=None):
        with torch.no_grad():
            weights0 = [
                net0.layers[layer_i].weight.clone().cpu() for i, layer_i in enumerate(self.layer_indices)
            ]
            weights1 = [
                net1.layers[layer_i].weight.clone().cpu() for i, layer_i in enumerate(self.layer_indices)
            ]

            perm_mats = [None for _ in range(len(weights0))]

            for i in range(len(weights0)):
                perm_mats[i] = torch.eye(
                    weights0[i].shape[0], 
                    weights0[i].shape[0]
                ).cpu()

            bestL = 0.0
            best_perm_mats = copy.deepcopy(perm_mats)
            
            for _ in range(epochs):
                oldL = torch.zeros(len(weights0))
                new_perm_mats = copy.deepcopy(perm_mats)

                newL = 0.0
                prevL = 0.0
                for i in range(len(weights0)):
                    if i == 0:
                        back = torch.eye(weights0[i].shape[1])
                    else:
                        back = new_perm_mats[i - 1].T
                    newL += torch.tensordot(weights0[i], new_perm_mats[i] @ weights1[i] @ back, dims=2)

                while True:
                    for i in torch.randperm(len(weights0)):
                        obj = torch.zeros(
                            weights0[i].shape[0], 
                            weights0[i].shape[0]
                        )
                        if i > 0:
                            obj = weights0[i] @ new_perm_mats[i - 1] @ weights1[i].T 

                        if i < len(weights0) - 1:
                            obj += weights0[i + 1].T @ new_perm_mats[i + 1] @ weights1[i + 1]
                        
                        new_perm_mat = self.perm_to_permmat(self.solve_lap(obj))

                        if i == 0:
                            back = torch.eye(weights0[i].shape[1])
                        else:
                            back = new_perm_mats[i - 1].T
                        L = torch.tensordot(weights0[i], new_perm_mat @ weights1[i] @ back, dims=2)
                        
                        if L > oldL[i]:
                            oldL[i] = L
                            new_perm_mats[i] = new_perm_mat

                    newL = 0.0
                    for i in range(len(weights0)):
                        if i == 0:
                            back = torch.eye(weights0[i].shape[1])
                        else:
                            back = new_perm_mats[i - 1].T
                        newL += torch.tensordot(weights0[i], new_perm_mats[i] @ weights1[i] @ back , dims=2)

                    print("DBG", newL)
                    if torch.abs(prevL - newL) < 10e-6:
                        break

                    prevL = newL

                    if newL > bestL:
                        best_perm_mats = copy.deepcopy(new_perm_mats)
                        bestL = newL

            print(bestL)
            return [self.permmat_to_perm(best_perm_mats[i].long()) for i in range(len(new_perm_mats))]

    def index_layers(self, model):
        if self.layers_indexed is True:
            return
        
        for idx, m in enumerate(model.layers):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                self.layer_indices.append(idx)

        self.layers_indexed = True

    def align_networks_smart(self, model0, model1, loader=None, device=None):
        cl0 = copy.deepcopy(model0.to("cpu")).to(device)
        cl1 = copy.deepcopy(model1.to("cpu")).to(device)

        self.index_layers(model0)

        if self.perms_calc is False:
            self.permutations = self.perm_coord_descent(cl0, cl1, epochs=200, device=device)
            self.perms_calc = True
        
        last_perm_map = None
        for i, layer_idx in enumerate(self.layer_indices):
            # if self.perms_calc is False:
            #     perm_map_, corr_mtx = self.get_layer_perm(
            #         model0.subnet(cl0, layer_i=layer_idx), 
            #         model1.subnet(cl1, layer_i=layer_idx), 
            #         epochs=1, 
            #         loader=loader, 
            #         device=device
            #     )
            #     self.permutations.append(perm_map_)

            perm_map = self.permutations[i]
            weight   = model1.layers[layer_idx].weight
            bias     = model1.layers[layer_idx].bias

            model1.layers[layer_idx].weight.data = weight[perm_map].clone()
            model1.layers[layer_idx].bias.data   = bias[perm_map].clone()

            weight = model1.layers[layer_idx].weight

            if i > 0:
                model1.layers[layer_idx].weight.data = weight[:, last_perm_map].clone()

            last_perm_map = perm_map

        last_perm_map = self.permutations[-1]
        weight = model1.classifier.weight

        model1.classifier.weight.data = weight[:, last_perm_map].clone()

        self.perms_calc = True

        return model0, model1

    def wrap_layers_smart(self, model, rescale):
        wrapped_model = model
        
        for i, layer_idx in enumerate(self.layer_indices):
            layer = model.layers[layer_idx]

            wrapper = LayerWrapper
            if isinstance(layer, nn.Conv2d):
                wrapper = LayerWrapper2D

            wrapped_model.layers[layer_idx] = wrapper(layer, rescale=rescale)

        return wrapped_model


    def mix_weights(self, model, model0, model1, alpha):
        sd0 = model0.state_dict()
        sd1 = model1.state_dict()
        sd_alpha = {k: (1 - alpha) * sd0[k] + alpha * sd1[k]
                    for k in sd0.keys()}
        
        model.load_state_dict(sd_alpha)
        

    def fuse_networks(self, model_args, model0, model1, alpha, loader=None, device=None, new_stats=True, permute = True):    
        modela = self.model_cls(**model_args).to(device)

        if permute is True:
            model0, model1 = self.align_networks_smart(
                model0, 
                model1, 
                loader=self.loaderc, 
                device=device
            )
            self.dbg0 = model0

        self.mix_weights(modela, model0, model1, alpha)

        if new_stats is False:
            return modela
        
        return self.REPAIR_smart(alpha, model0, model1, modela, loader=loader, device=device)
    
    def REPAIR_smart(self, alpha, model0, model1, modela, loader=None, device=None):
        model0_tracked = self.wrap_layers_smart(model0, rescale=False).to(device)
        model1_tracked = self.wrap_layers_smart(model1, rescale=False).to(device)
        modela_tracked = self.wrap_layers_smart(modela, rescale=False).to(device)

        if self.stats_calc is False:
            model0_tracked.train()
            model1_tracked.train()
            for m in model0_tracked.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.momentum = None
                    m.reset_running_stats()

            for m in model1_tracked.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.momentum = None
                    m.reset_running_stats()
            
            with torch.no_grad():
                for inputs, labels in self.loader0:
                    _ = model0_tracked(inputs.to(device))

                for inputs, labels in self.loader1:
                    _ = model1_tracked(inputs.to(device))

            model0_tracked.eval()
            model1_tracked.eval()
            
            for i, layer_idx in enumerate(self.layer_indices):
                layer0 = model0_tracked.layers[layer_idx]
                layer1 = model1_tracked.layers[layer_idx]

                stats0_ = layer0.get_stats()
                stats1_ = layer1.get_stats()

                self.statistics.append((stats0_, stats1_))

        for i, layer_idx in enumerate(self.layer_indices):
            stats0 = self.statistics[i][0]
            stats1 = self.statistics[i][1]

            mean = (1.0 - alpha) * stats0[0] + alpha * stats1[0]
            std = ((1.0 - alpha) * stats0[1].sqrt() + alpha * stats1[1].sqrt()).square()
    
            modela_tracked.layers[layer_idx].set_stats(mean, std)
            modela_tracked.layers[layer_idx].rescale = True

        modela_tracked.train()
        for m in modela_tracked.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.momentum = None
                m.reset_running_stats()
        
        with torch.no_grad():
            for _ in range(3):
                for inputs, labels in self.loaderc:
                    _ = modela_tracked(inputs.to(device))

        modela_tracked.eval()
        
        self.stats_calc = True

        return modela_tracked
    