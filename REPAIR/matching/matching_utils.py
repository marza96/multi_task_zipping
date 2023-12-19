import torch
import scipy
import copy

import numpy as np


def apply_permutation(layer_indices, net, perms, device=None):
    last_perm_map   = None
    net             = copy.deepcopy(net).to(device)
    
    for i, layer_idx in enumerate(layer_indices):
        perm_map = perms[i]
        weight   = net.layers[layer_idx].weight
        bias     = net.layers[layer_idx].bias

        net.layers[layer_idx].weight.data = weight[perm_map].clone()
        net.layers[layer_idx].bias.data   = bias[perm_map].clone()

        weight = net.layers[layer_idx].weight
        
        if i > 0:
            net.layers[layer_idx].weight.data = weight[:, last_perm_map].clone()

        last_perm_map = perm_map

    return net


def perm_to_permmat(permutation):
    perm_mat = torch.zeros((len(permutation), len(permutation))).to(permutation.device)
    perm_mat[torch.arange(len(permutation)), permutation] = 1

    return perm_mat


def permmat_to_perm(permmat):
    perm = torch.Tensor(permmat.shape[0]).to(permmat.device)
    perm = torch.where(permmat == 1)[1]

    return perm


def solve_lap(corr_mtx, maximize=True):
    corr_mtx_a = corr_mtx.cpu().numpy()
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(corr_mtx_a, maximize=maximize)

    assert (row_ind == np.arange(len(corr_mtx_a))).all()

    perm_map = torch.tensor(col_ind).long()

    return perm_map

