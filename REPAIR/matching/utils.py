import torch
import scipy

import numpy as np


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