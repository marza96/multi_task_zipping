import copy
import tqdm
import torch

from .weight_matching import WeightMatching
from .matching_utils import apply_permutation, perm_to_permmat, permmat_to_perm


class PGDMatching:
    def __init__(self, loss_fn, loader, lr, debug=False, epochs=6, ret_perms=False, device="mps", wm_kwargs=None):
        self.debug     = debug
        self.loader    = loader
        self.epochs    = epochs
        self.device    = device
        self.loss_fn   = loss_fn
        self.ret_perms = ret_perms
        self.lr        = lr

        if wm_kwargs is None:
            wm_kwargs = dict()

        self.weight_matching = WeightMatching(**wm_kwargs)
        self.final_perm      = list()
        self.netm            = None

    def mix_perm_weights(self, model0, model1, perms, layer_indices, alpha=0.5, device=None):
        model = copy.deepcopy(model0.cpu()).to(device)
        model1_new = copy.deepcopy(model1.cpu()).to(device)

        p_in  = None
        p_out = None

        for idx, layer_idx in enumerate(layer_indices):
            w = model1_new.layers[layer_idx].weight.clone()
            b = model1_new.layers[layer_idx].bias.clone()

            if idx == 0:
                p_in = torch.arange(w.shape[1]).long()
                p_in.to(device)
            elif idx > 0:
                p_in = perms[idx - 1]
                p_in.to(device)

            if idx < len(perms):
                p_out = perms[idx]
                p_out.to(device)
            else:
                p_out = torch.arange(w.shape[0])
                p_out.to(device)

            w = w[p_out, :]
            w = w[:, p_in]
            b = b[p_out] 

            model1_new.layers[layer_idx].weight.data = w
            model1_new.layers[layer_idx].bias.data = b

            w_0 = model0.layers[layer_idx].weight.data.to(device)
            b_0 = model0.layers[layer_idx].bias.data.to(device)

            model.layers[layer_idx].weight.data = alpha * w + (1.0 - alpha) * w_0
            model.layers[layer_idx].bias.data = alpha * b + (1.0 - alpha) * b_0

        return model, model1_new
    
    
    def learnable(self, layer_indices, net0, net1, debug_perms=None, init_perm=None):
        perms = None

        net0 = copy.deepcopy(net0)
        netm = copy.deepcopy(net1)
        net1 = copy.deepcopy(net1)

        for _, param in net1.named_parameters():
            param.requires_grad = False

        for _, param in net0.named_parameters():
            param.requires_grad = False

        for iteration in range(10): #10000
            cost = torch.zeros((1))
            
            netm.to(self.device)
            net0.to(self.device)
            net1.to(self.device)

            l = 0.0
            c = 0

            for i, (images, labels) in enumerate(self.loader):
                for _, param in netm.named_parameters():
                    param.requires_grad = True

                for _, param in net1.named_parameters():
                    param.requires_grad = False

                for _, param in net0.named_parameters():
                    param.requires_grad = False

                netf = copy.deepcopy(netm).to(self.device)
                for layer_idx in layer_indices:
                    wm = netm.layers[layer_idx].weight
                    bm = netm.layers[layer_idx].bias

                    w0 = net0.layers[layer_idx].weight
                    b0 = net0.layers[layer_idx].bias

                    netf.layers[layer_idx].weight.data = 0.5 * (wm + w0)
                    netf.layers[layer_idx].bias.data = 0.5 * (bm + b0)

                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = netf(images)
                loss = torch.nn.functional.cross_entropy(outputs, labels) 
                gradient = torch.autograd.grad(loss, netf.parameters())

                for t, grad, n0 in zip(netm.named_parameters(), gradient, net0.parameters()):
                    param = t[1]

                    new_param = param.detach() - 0.05 * grad.detach()
                    param.data = new_param.detach()
                
                l += loss.mean()
                c += 1

            print("loss", l / c)

            if iteration % 5 == 0:
                perms = self.weight_matching(layer_indices, copy.deepcopy(netm), copy.deepcopy(net1), penalty=10000 * torch.eye(128))
                net1 = apply_permutation(layer_indices, net1, perms)
                netm = copy.deepcopy(net1)

                cost = torch.zeros((1)).to(self.device)
                for layer_idx in layer_indices:
                    cost += torch.norm(net1.layers[layer_idx].weight - net0.layers[layer_idx].weight)
                    cost += torch.norm(net1.layers[layer_idx].bias - net0.layers[layer_idx].bias)

                print("COST %d" % iteration, cost)

        return net0, net1
    
    def acumulate_perms(self, perms):
        for idx, perm in enumerate(perms):
            if idx == len(perms) - 1:
                break
                
            self.final_perm[idx] = perm_to_permmat(perm) @ self.final_perm[idx]
        
    def __call__(self, layer_indices, net0, net1, debug_perms=None, init_perm=None):
        perms      = None
        net0       = copy.deepcopy(net0)
        netm       = copy.deepcopy(net1)
        net1       = copy.deepcopy(net1)

        for i in range(len(layer_indices) - 1):
            perm_shape = net0.layers[layer_indices[i]].weight.shape[0]
            self.final_perm.append(perm_to_permmat(torch.arange(perm_shape)))

        for _, param in net1.named_parameters():
            param.requires_grad = False

        for _, param in net0.named_parameters():
            param.requires_grad = False

        costs = list()
        for iteration in range(self.epochs):
            for _, param in netm.named_parameters():
                param.requires_grad = True

            cost = torch.zeros((1))
            for layer_idx in layer_indices:
                cost += torch.norm(netm.layers[layer_idx].weight - net0.layers[layer_idx].weight)
                cost += torch.norm(netm.layers[layer_idx].bias - net0.layers[layer_idx].bias)
            
            gradient = torch.autograd.grad(cost, netm.parameters())
            
            for t, grad, n0 in zip(netm.named_parameters(), gradient, net0.parameters()):
                param = t[1]

                new_param = param.detach() - grad.detach().cpu()
                param.data = new_param.detach()
           
            perms = self.weight_matching(layer_indices, copy.deepcopy(netm), copy.deepcopy(net1))
            dummy = apply_permutation(layer_indices, net1, perms)

            cost = torch.zeros((1))
            for layer_idx in layer_indices:
                cost += torch.norm(dummy.layers[layer_idx].weight - net0.layers[layer_idx].weight)
                cost += torch.norm(dummy.layers[layer_idx].bias - net0.layers[layer_idx].bias)
            
            cost2 = torch.zeros((1))
            for layer_idx in layer_indices:
                cost2 += torch.norm(net1.layers[layer_idx].weight - net0.layers[layer_idx].weight)
                cost2 += torch.norm(net1.layers[layer_idx].bias - net0.layers[layer_idx].bias)
            
            if cost < cost2:
                perms = self.weight_matching(layer_indices, copy.deepcopy(netm), copy.deepcopy(net1))
                net1 = apply_permutation(layer_indices, net1, perms)
                netm = copy.deepcopy(net1)

                self.acumulate_perms(perms)

                cost = torch.zeros((1))
                for layer_idx in layer_indices:
                    cost += torch.norm(netm.layers[layer_idx].weight - net0.layers[layer_idx].weight)
                    cost += torch.norm(netm.layers[layer_idx].bias - net0.layers[layer_idx].bias)

                costs.append(float(cost))
                print("COST %d" % iteration, cost)

        if self.ret_perms is True:
            return [permmat_to_perm(perm) for perm in self.final_perm]
        
        return net0, net1
