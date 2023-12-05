import torch
import scipy
import copy

import numpy as np


def apply_permutation(layer_indices, net, perms):
    last_perm_map   = None
    net             = copy.deepcopy(net)
    last_perm_shape = net.layers[layer_indices[-1]].weight.shape[1]
    
    perms.append(permmat_to_perm(torch.eye(last_perm_shape)))

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
    perm_mat = torch.zeros((len(permutation), len(permutation)))
    perm_mat[torch.arange(len(permutation)), permutation] = 1

    return perm_mat


def permmat_to_perm(permmat):
    perm = torch.Tensor(permmat.shape[0])
    perm = torch.where(permmat == 1)[1]

    return perm


def solve_lap(corr_mtx):
    corr_mtx_a = corr_mtx.cpu().numpy()
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(corr_mtx_a, maximize=True)

    assert (row_ind == np.arange(len(corr_mtx_a))).all()

    perm_map = torch.tensor(col_ind).long()

    return perm_map