import numpy as np
import torch
import random

def perm_to_permmat(permutation):
    perm_mat = torch.zeros((5, 5))
    perm_mat[torch.arange(5), permutation] = 1

    return perm_mat

def permmat_to_perm(permmat):
    perm = torch.Tensor(permmat.shape[0])
    perm = torch.where(permmat == 1)[1]

    return perm

def main():
    permutation = list(range(5))
    random.shuffle(permutation)

    print(permutation)
    
    perm_mat = perm_to_permmat(permutation)

    print(perm_mat)

    print(np.hstack((perm_mat, np.arange(5).reshape(-1, 1))))

    perm = permmat_to_perm(perm_mat)
    print(perm)

if __name__ == "__main__":
    main()