from typing import Any
import torch
import copy

from .matching_utils import perm_to_permmat, permmat_to_perm, solve_lap, apply_permutation


class WeightMatching():
    def __init__(self, debug=False, epochs=1, debug_perms=None, ret_perms=False, device="cpu"):
        self.debug       = debug
        self.epochs      = epochs
        self.debug_perms = debug_perms
        self.ret_perms   = ret_perms
        self.device      = device

    def linear_objective(self, idx, perm_mats, weights0, weights1, biases0, biases1, l_types):
        obj = torch.zeros(
            (weights0[idx].shape[0], weights0[idx].shape[0])
        )

        w0_i = weights0[idx].clone()
        w1_i = weights1[idx].clone()

        if biases0[idx] is not None:
            obj += torch.outer(biases0[idx], biases1[idx])

        if idx > 0:
            obj += w0_i @ perm_mats[idx - 1] @ w1_i.T 

        if idx == 0:
            obj += w0_i @  w1_i.T 

        if idx < len(weights0) - 1:
            w0_ii = weights0[idx + 1].clone()
            w1_ii = weights1[idx + 1].clone()

            obj += w0_ii.T @ perm_mats[idx + 1] @ w1_ii
        
        return obj


    def conv2d_objective(self, idx, perm_mats, weights0, weights1, biases0, biases1, l_types):
        kernel_shape = weights0[0].shape[2:]
        obj          = torch.zeros(
            (
                *kernel_shape, 
                weights0[idx].shape[0], 
                weights0[idx].shape[0]
            )
        )

        w0_i = weights0[idx].clone().permute(2, 3, 0, 1)
        w1_i = weights1[idx].clone().permute(2, 3, 0, 1)

        if biases0[idx] is not None:
            obj += torch.outer(biases0[idx], biases1[idx])

        if idx > 0:
            p = torch.empty(
                kernel_shape[0] * kernel_shape[1], 
                *(perm_mats[idx - 1].shape)
            )

            for i in range(kernel_shape[0] * kernel_shape[1]):
                p[i, :, :] = perm_mats[idx - 1].clone()

            p = p.reshape(*kernel_shape, *(p.shape[1:]))

            obj += w0_i @ p @ w1_i.permute(0, 1, 3, 2) 

        if idx == 0:
            obj += w0_i @ w1_i.permute(0, 1, 3, 2)

        if idx < len(weights0) - 1:
            w0_ii = weights0[idx + 1].clone()
            w1_ii = weights1[idx + 1].clone()

            if len(w0_ii.shape) == 2:
                w0_ii = w0_ii[:, :, None, None]
                w1_ii = w1_ii[:, :, None, None]

            w0_ii = w0_ii.permute(2, 3, 0, 1)
            w1_ii = w1_ii.permute(2, 3, 0, 1)

            p = torch.empty(
                kernel_shape[0] * kernel_shape[1], 
                *(perm_mats[idx + 1].shape)
            )
            
            for i in range(kernel_shape[0] * kernel_shape[1]):
                p[i, :, :] = perm_mats[idx + 1].clone()

            p = p.reshape(*kernel_shape, *(p.shape[1:]))

            obj += w0_ii.permute(0, 1, 3, 2) @ p @ w1_ii
        
        return obj.sum(0).sum(0)

    def objective(self, idx, perm_mats, weights0, weights1, biases0, biases1, l_types):
        if isinstance(l_types[idx], torch.nn.Linear):
            return self.linear_objective(
                idx, perm_mats, weights0, weights1, biases0, biases1, l_types)

        if isinstance(l_types[idx], torch.nn.Conv2d):
            return self.conv2d_objective(
                idx, perm_mats, weights0, weights1, biases0, biases1, l_types)
    
    def apply_permutation(self, layer_indices, net, perms):
        last_perm_map = None
        net = copy.deepcopy(net)
        perms =  perms

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

    def __call__(self, layer_indices, net0, net1, ste=False, init_perm=None):
        net0.to("cpu")
        net1.to("cpu")

        with torch.no_grad():
            if ste is True:
                weights0 = [
                    net0.layers[layer_i].layer_hat.weight.clone().cpu().float() for i, layer_i in enumerate(layer_indices) #if i < len(layer_indices) - 1
                ]
                weights1 = [
                    net1.layers[layer_i].weight.clone().cpu().float() for i, layer_i in enumerate(layer_indices) #if i < len(layer_indices) - 1
                ]
                biases0 = [
                    net0.layers[layer_i].layer_hat.bias.clone().cpu().float() for i, layer_i in enumerate(layer_indices) #if i < len(layer_indices) - 1
                ]
                biases1 = [
                    net1.layers[layer_i].bias.clone().cpu().float() for i, layer_i in enumerate(layer_indices) #if i < len(layer_indices) - 1
                ]
                l_types = [
                    net1.layers[layer_i] for i, layer_i in enumerate(layer_indices) #if i < len(layer_indices) - 1
                ]
            else:
                weights0 = [
                    net0.layers[layer_i].weight.clone().cpu().float() for i, layer_i in enumerate(layer_indices) #if i < len(layer_indices) - 1
                ]
                weights1 = [
                    net1.layers[layer_i].weight.clone().cpu().float() for i, layer_i in enumerate(layer_indices) #if i < len(layer_indices) - 1
                ]
                biases0 = [
                    net0.layers[layer_i].bias.clone().cpu().float() for i, layer_i in enumerate(layer_indices) #if i < len(layer_indices) - 1
                ]
                biases1 = [
                    net1.layers[layer_i].bias.clone().cpu().float() for i, layer_i in enumerate(layer_indices) #if i < len(layer_indices) - 1
                ]
                l_types = [
                    net1.layers[layer_i] for i, layer_i in enumerate(layer_indices) #if i < len(layer_indices) - 1
                ]

            perm_mats = [None for _ in range(len(weights0))]

            for i in range(len(weights0)):
                perm_mats[i] = torch.eye(
                    weights0[i].shape[0]
                ).cpu()

            if init_perm is not None:
                for i in range(len(weights0) - 1):
                    perm_mats[i] = perm_to_permmat(init_perm[i])
            
            for iteration in range(self.epochs):
                progress = False
                rperm = torch.randperm(len(layer_indices) - 1)

                if self.debug_perms is not None:
                    rperm = self.debug_perms[iteration]

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
            
            final_perms = [permmat_to_perm(perm_mats[i].long()) for i in range(len(perm_mats))]

            if self.ret_perms is True:
                return final_perms
            
            final_perms += [permmat_to_perm(torch.eye(weights0[-1].shape[0]))]
            net1 = self.apply_permutation(layer_indices, net1, final_perms)
            
            return net0.to(self.device), net1.to(self.device)
        