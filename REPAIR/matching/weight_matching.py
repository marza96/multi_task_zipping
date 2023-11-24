import torch
import scipy
import tqdm
import copy

from .utils import perm_to_permmat, permmat_to_perm, solve_lap


class WeightMatching():
    def __init__(self, debug=False):
        self.debug = debug

    def objective(self, idx, perm_mats, weights0, weights1, biases0, biases1, l_types):
        obj = torch.zeros(
            (weights0[idx].shape[0], weights0[idx].shape[0])
        )

        w0_i = weights0[idx].clone()
        w1_i = weights1[idx].clone()

        if isinstance(l_types[idx], torch.nn.Conv2d):
            w0_i = w0_i.permute(2, 3, 0, 1)
            w1_i = w1_i.permute(2, 3, 0, 1)

        if biases0[idx] is not None:
            obj += torch.outer(biases0[idx], biases1[idx])

        if idx > 0:
            obj += w0_i @ perm_mats[idx - 1] @ w1_i.T 

        if idx == 0:
            obj += w0_i @  w1_i.T 

        if idx < len(weights0) - 1:
            w0_ii = weights0[idx + 1].clone()
            w1_ii = weights1[idx + 1].clone()

            if isinstance(l_types[idx], torch.nn.Conv2d):
                w0_ii = w0_ii.permute(2, 3, 0, 1)
                w1_ii = w1_ii.permute(2, 3, 0, 1)

            obj += w0_ii.T @ perm_mats[idx + 1] @ w1_ii

        return obj

    def __call__(self, layer_indices, net0, net1, epochs=2):
        with torch.no_grad():
            weights0 = [
                net0.layers[layer_i].weight.clone().cpu() for i, layer_i in enumerate(layer_indices)
            ]
            weights1 = [
                net1.layers[layer_i].weight.clone().cpu() for i, layer_i in enumerate(layer_indices)
            ]
            biases0 = [
                net0.layers[layer_i].bias.clone().cpu() for i, layer_i in enumerate(layer_indices)
            ]
            biases1 = [
                net1.layers[layer_i].bias.clone().cpu() for i, layer_i in enumerate(layer_indices)
            ]
            l_types = [
                type(net1.layers[layer_i]) for i, layer_i in enumerate(layer_indices)
            ]

            perm_mats = [None for _ in range(len(weights0))]

            for i in range(len(weights0)):
                perm_mats[i] = torch.eye(
                    weights0[i].shape[0]
                ).cpu()

            for iteration in range(epochs):
                progress = False
                rperm = torch.randperm(len(weights0))

                if self.debug is True:
                    rperm = torch.Tensor([4, 1, 3, 2, 0]).long()
                
                for i in rperm:
                    obj  = self.objective(i, perm_mats, weights0, weights1, biases0, biases1, l_types)
                    perm = solve_lap(obj)
                    oldL = torch.einsum(
                        'ij,ij->i', 
                        obj, 
                        torch.eye(weights0[i].shape[0])[permmat_to_perm(perm_mats[i]).long(), :]
                    ).sum()
                    newL = torch.einsum(
                        'ij,ij->i', 
                        obj, 
                        torch.eye(weights0[i].shape[0])[perm, :]
                    ).sum()
                
                    progress     = progress or newL > oldL + 1e-12
                    perm_mats[i] = copy.deepcopy(perm_to_permmat(perm))

                    if self.debug is True:
                        print(f"{iteration}/{i}: {newL - oldL}") 

                if not progress:
                    break

            return [permmat_to_perm(perm_mats[i].long()) for i in range(len(perm_mats))]
        