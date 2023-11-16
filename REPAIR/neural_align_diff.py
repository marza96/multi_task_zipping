import torch
import scipy
import copy

from tqdm import tqdm

from net_models.mlp import LayerWrapper, MLP

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
    def __init__(self, loader0, loader1, loaderc) -> None:
        self.permutations = list()
        self.statistics   = list()
        self.perms_calc   = False
        self.stats_calc   = False

        self.loader0 = loader0
        self.loader1 = loader1
        self.loaderc = loaderc

        self.dbg0 = None

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

    def optimize_corr(self, corr_mtx):
        corr_mtx_a = corr_mtx.cpu().numpy()
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(corr_mtx_a, maximize=True)

        assert (row_ind == np.arange(len(corr_mtx_a))).all()

        perm_map = torch.tensor(col_ind).long()

        return perm_map
    
    def perm_to_permmat(self, permutation):
        perm_mat = torch.zeros((5, 5))
        perm_mat[torch.arange(5), permutation] = 1

        return perm_mat

    def permmat_to_perm(permmat):
        perm = torch.Tensor(permmat.shape[0])
        perm = torch.where(permmat == 1)[1]

        return perm

    def get_layer_perm(self, net0, net1, epochs=1, loader=None, device=None):
        corr_mtx = self.run_corr_matrix(net0, net1, epochs=1, loader=loader, device=device)

        return self.optimize_corr(corr_mtx), corr_mtx
    
    def perm_coord_descent(self, net0, net1, epochs=1, loader=None, device=None):
        perm_mats = [None for _ in range(net0.num_layers + 2)]
        perm_mats[0] = torch.eye(net0.fc1.weight.shape[1], net0.fc1.weight.shape[1])

        for i in range(net0.num_layers):
            perm_mats[i + 1] = torch.eye(net0.layers[2 * i].weight.shape[1], net0.layers[2 * i].weight.shape[1])

        perm_mats[-1] = torch.eye(net0.fc2.weight.shape[1], net0.fc2.weight.shape[1])
        
        w_0 = net0.fc1.weight.clone()
        w_1 = net0.layers[0].weight.clone()
        obj = w_0 @ perm_mats[0] @ w_0.T + w_1.T @ perm_mats[1] @ w_1
        
        w_i = w_1
        for i in range(net0.num_layers):
            if i < net0.num_layers - 1:
                w_ii = net0.layers[2 * (i + 1)].weight.clone()
            else:
                w_ii = net0.fc2.weight.clone()
            
            obj = w_i @ perm_mats[i + 1] @ w_i.T 

            if i < net0.num_layers - 1:
                obj += w_ii.T @ perm_mats[i + 2] @ w_ii

            w_i = w_ii  

    def geet_layer_perm_ste(self, net0, net1, epochs=1, loader=None, device=None):
        pass

    def align_networks(self, model0, model1, layers, loader=None, device=None):
        cl0 = copy.deepcopy(model0.to("cpu")).to(device)
        cl1 = copy.deepcopy(model1.to("cpu")).to(device)

        if self.perms_calc is False:
            perm_map_, corr_mtx = self.get_layer_perm(model0.subnet(cl0, layer_i=0), model1.subnet(cl1, layer_i=0), epochs=1, loader=loader, device=device)
            self.permutations.append(perm_map_)
        
        perm_map = self.permutations[0]
        weight   = model1.fc1.weight
        bias     = model1.fc1.bias

        model1.fc1.weight.data = weight[perm_map].clone()
        model1.fc1.bias.data   = bias[perm_map].clone()

        last_perm_map = perm_map

        for i in range(layers):
            if self.perms_calc is False:
                perm_map_, corr_mtx = self.get_layer_perm(model0.subnet(cl0, layer_i=i + 1), model1.subnet(cl1, layer_i=i + 1), epochs=1, loader=loader, device=device)
                self.permutations.append(perm_map_)

            perm_map = self.permutations[i + 1]
            weight   = model1.layers[2 * i].weight
            bias     = model1.layers[2 * i].bias  

            model1.layers[2 * i].weight.data = weight[perm_map].clone()
            model1.layers[2 * i].bias.data   = bias[perm_map].clone()

            weight   = model1.layers[2 * i].weight

            model1.layers[2 * i].weight.data = weight[:, last_perm_map].clone()
            last_perm_map = perm_map
        
        weight = model1.fc2.weight

        model1.fc2.weight.data = weight[:, last_perm_map]
        self.perms_calc = True

        return model0, model1

    def wrap_layers(self, model, rescale):
        wrapped_model = model

        wrapped_model.fc1 = LayerWrapper(wrapped_model.fc1, rescale=rescale)

        for i in range(len(wrapped_model.layers)):
            layer = wrapped_model.layers[i]

            if isinstance(layer, nn.Linear):
                wrapped_model.layers[i] = LayerWrapper(wrapped_model.layers[i], rescale=rescale)

        return wrapped_model


    def mix_weights(self, model, model0, model1, alpha):
        sd0 = model0.state_dict()
        sd1 = model1.state_dict()
        sd_alpha = {k: (1 - alpha) * sd0[k] + alpha * sd1[k]
                    for k in sd0.keys()}
        
        model.load_state_dict(sd_alpha)
        

    def fuse_networks(self, model0, model1, alpha, layers, loader=None, device=None, new_stats=True, permute = True):    
        modela = MLP(model0.h, model0.num_layers).to(device)

        if permute is True:
            model0, model1 = self.align_networks(
                model0, 
                model1, 
                layers, 
                loader=self.loaderc, 
                device=device
            )
            self.dbg0 = model0

        self.mix_weights(modela, model0, model1, alpha)

        if new_stats is False:
            return modela
        
        return self.REPAIR(alpha, model0, model1, modela, loader=loader, device=device)
        
    def REPAIR(self, alpha, model0, model1, modela, loader=None, device=None):
        model0_tracked = self.wrap_layers(model0, rescale=False).to(device)
        model1_tracked = self.wrap_layers(model1, rescale=False).to(device)
        modela_tracked = self.wrap_layers(modela, rescale=False).to(device)

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
                    o2 = model0_tracked(inputs.to(device))

                for inputs, labels in self.loader1:
                    o1 = model1_tracked(inputs.to(device))

            model0_tracked.eval()
            model1_tracked.eval()
            
            stats0_ = model0_tracked.fc1.get_stats()
            stats1_ = model1_tracked.fc1.get_stats()

            self.statistics.append((stats0_, stats1_))

            for i in range(len(model0.layers)):
                if not isinstance(model0_tracked.layers[i], LayerWrapper):
                    continue

                stats0_ = model0_tracked.layers[i].get_stats()
                stats1_ = model1_tracked.layers[i].get_stats()

                self.statistics.append((stats0_, stats1_))

        stats0 = self.statistics[0][0]
        stats1 = self.statistics[0][1]

        mean = (1.0 - alpha) * stats0[0] + alpha * stats1[0]
        std = ((1.0 - alpha) * stats0[1].sqrt() + alpha *stats1[1].sqrt()).square()
        
        modela_tracked.fc1.set_stats(mean, std)
        modela_tracked.fc1.rescale = True

        cnt = 1
        for i in range(len(model0.layers)):
            if not isinstance(model0_tracked.layers[i], LayerWrapper):
                continue

            # stats0 = model0_tracked.layers[i].get_stats()
            # stats1 = model1_tracked.layers[i].get_stats()

            stats0 = self.statistics[cnt][0]
            stats1 = self.statistics[cnt][1]

            mean = (1.0 - alpha) * stats0[0] + alpha * stats1[0]
            std = ((1.0 - alpha) * stats0[1].sqrt() + alpha * stats1[1].sqrt()).square()
    
            modela_tracked.layers[i].set_stats(mean, std)
            modela_tracked.layers[i].rescale = True

            cnt += 1
        
        modela_tracked.train()
        for m in modela_tracked.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.momentum = None
                m.reset_running_stats()
        
        with torch.no_grad():
            for _ in range(3):
                for inputs, labels in self.loaderc:
                    o1 = modela_tracked(inputs.to(device))

        modela_tracked.eval()
        
        self.stats_calc = True

        return modela_tracked
        
