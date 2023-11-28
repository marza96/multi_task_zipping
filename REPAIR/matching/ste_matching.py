import copy
import tqdm
import torch
import torch.nn.functional as F

from torch import Tensor
from torch.nn.modules.module import Module
from torch.optim import SGD

from .weight_matching import WeightMatching

from torch.nn.utils.stateless import functional_call

from collections import OrderedDict
from torchviz import make_dot
# import torchopt


class ModSTE(torch.autograd.Function):
    @staticmethod
    def forward(input, w_for, b_for, w_back, b_back, w_0):
        a = input.mm(w_for.detach().t()) + b_for.detach()

        return a 

    @staticmethod
    def setup_context(ctx, inputs, output):
        input, w_for, b_for, w_back, b_back, w_0 = inputs
        ctx.save_for_backward(input, w_for, b_for, w_back, b_back, w_0)

    @staticmethod
    def backward(ctx, grad_output):
        input, w_for, b_for, w_back, b_back, w_0 = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        grad_bias   = grad_output.sum(0) * 0.5
        grad_input  = grad_output.mm(w_0.detach())
        grad_weight = grad_output.t().mm(input) * 0.5

        return grad_input, None, None, grad_weight, grad_bias, None


class ModLinearStraightThroughEstimator(torch.nn.Module):
    def __init__(self, layer_0, layer_1, idx, perms):
        super().__init__()
        
        self.idx   = idx
        self.perms = perms
        self.w_0   = layer_0.weight.detach().float()
        self.w_1   = layer_1.weight.detach().float()
        self.b_0   = layer_0.bias.detach().float()
        self.b_1   = layer_1.bias.detach().float()

        self.weight = torch.nn.Parameter(layer_0.weight.detach().float())
        self.bias   = torch.nn.Parameter(layer_0.bias.detach().float())

        self.weight.requires_grad = True
        self.bias.requires_grad   = True

    def reset(self, layer_1, perms):
        self.weight.requires_grad = True
        self.bias.requires_grad   = True

        self.w_1   = layer_1.weight.detach().float()
        self.b_1   = layer_1.bias.detach().float()
        self.perms = perms

    def forward(self, input): 
        x     = input
        p_in  = None
        p_out = None

        if self.idx == 0:
            p_in = torch.arange(x.shape[1]).long()
            p_in.to(x.device)
        elif self.idx > 0:
            p_in = self.perms[self.idx - 1]
            p_in.to(x.device)

        if self.idx < len(self.perms):
            p_out = self.perms[self.idx]
            p_out.to(x.device)
        else:
            p_out = torch.arange(self.weight.shape[0])
            p_out.to(x.device)

        self.w_1 = self.w_1[p_out, :]
        self.w_1 = self.w_1[:, p_in]
        self.b_1 = self.b_1[p_out]

        x = ModSTE.apply(
            x, 
            0.5 * (self.w_0 + self.w_1),
            0.5 * (self.b_0 + self.b_1),
            self.weight,
            self.bias,
            0.5 * (self.weight.detach() + self.w_1)
        )
        
        return x
    




class STE(torch.autograd.Function):
    @staticmethod
    def forward(input, w_for, b_for, w_back, b_back):
        a = input.mm(w_for.detach().t()) + b_for.detach()

        return a 

    @staticmethod
    def setup_context(ctx, inputs, output):
        input, w_for, b_for, w_back, b_back = inputs
        ctx.save_for_backward(input, w_for, b_for, w_back, b_back)

    @staticmethod
    def backward(ctx, grad_output):
        input, w_for, b_for, w_back, b_back = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        grad_bias   = grad_output.sum(0)
        grad_input  = grad_output.mm(w_back.detach())
        grad_weight = grad_output.t().mm(input)

        return grad_input, None, None, grad_weight, grad_bias


class LinearStraightThroughEstimator(torch.nn.Module):
    def __init__(self, layer_0, layer_1, idx, perms):
        super().__init__()
        
        self.idx   = idx
        self.perms = perms
        self.w_0   = layer_0.weight.detach().float()
        self.w_1   = layer_1.weight.detach().float()
        self.b_0   = layer_0.bias.detach().float()
        self.b_1   = layer_1.bias.detach().float()

        self.weight = torch.nn.Parameter(layer_0.weight.detach().float())
        self.bias   = torch.nn.Parameter(layer_0.bias.detach().float())

        self.weight.requires_grad = True
        self.bias.requires_grad   = True

    def reset(self, layer_1, perms):
        self.weight.requires_grad = True
        self.bias.requires_grad   = True

        self.w_1   = layer_1.weight.detach().float()
        self.b_1   = layer_1.bias.detach().float()
        self.perms = perms

    def forward(self, input): 
        x     = input
        p_in  = None
        p_out = None

        if self.idx == 0:
            p_in = torch.arange(x.shape[1]).long()
            p_in.to(x.device)
        elif self.idx > 0:
            p_in = self.perms[self.idx - 1]
            p_in.to(x.device)

        if self.idx < len(self.perms):
            p_out = self.perms[self.idx]
            p_out.to(x.device)
        else:
            p_out = torch.arange(self.weight.shape[0])
            p_out.to(x.device)

        self.w_1 = self.w_1[p_out, :]
        self.w_1 = self.w_1[:, p_in]
        self.b_1 = self.b_1[p_out]

        x = STE.apply(
            x, 
            0.5 * (self.w_0 + self.w_1),
            0.5 * (self.b_0 + self.b_1),
            0.5 * (self.w_1 + self.weight),
            0.5 * (self.b_1 + self.bias),
        )
        
        return x
    

    

class SteMatching:
    def __init__(self, loss_fn, loader, lr, debug=False, epochs=6, device="mps", wm_kwargs=None):
        self.debug   = debug
        self.loader  = loader
        self.epochs  = epochs
        self.device  = device
        self.loss_fn = loss_fn
        self.lr      = lr

        if wm_kwargs is None:
            wm_kwargs = dict()

        self.weight_matching = WeightMatching(**wm_kwargs)

        self.netm = None

    def __call__(self, layer_indices, net0, net1, debug_perms=None):

        best_perm_loss = 1000.0
        best_perm = None
        perms = None

        netm = copy.deepcopy(net0)

        loss_fn = torch.nn.NLLLoss()
        netm = self._wrap_network(layer_indices, net0.to(self.device), net1.to(self.device), copy.deepcopy(perms))
        # optimizer = SGD(netm.parameters(), lr=0.01, momentum=0.0)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.75)
        # optimizer = torchopt.sgd(lr=0.5, momentum=0.9, moment_requires_grad=True)
        # opt_state = None

        perms = self.weight_matching(layer_indices, copy.deepcopy(netm), copy.deepcopy(net1), init_perm=perms)

        for iteration in range(self.epochs):
            loss_acum = 0.0
            total = 0

            for i, (images, labels) in enumerate(tqdm.tqdm(self.loader)):
                images = images.to(self.device)
                labels = labels.to(self.device)

                self._reset_network(layer_indices, netm, net1.to(self.device), copy.deepcopy(perms), zero_grad=True)
                # if opt_state is None:
                    # opt_state = optimizer.init(netm.state_dict())

                outputs = netm(images)
                print("OUT MY", outputs[0, :11])

                loss = loss_fn(outputs, labels)
                loss.backward()

                for perm in perms:
                        print(perm[:11])
                # if i == 1:
                #     print("iter %d ....................." % i)
                #     for name, param in netm.named_parameters():
                #         if param.requires_grad is False:
                #             continue
                        
                #         if not "layers.10" in name:
                #             continue
                        
                #         try:
                #             print("MY W", name, param[:5, :5])
                #         except:
                #             print("MY W", name, param[:5])
                #     print("......................................")
                #     return

                if i == 1: 
                    print("iter %d ....................." % i)
                    for name, param in netm.named_parameters():
                        if param.requires_grad is False:
                            continue

                        if not "layers.10" in name:
                            continue
                        
                        try:
                            print("MY G", name, param.grad[:5, :5])
                        except:
                            print("MY G", name, param.grad[:5])
                    print("......................................")

                # GOOD LR 0.00001
                for name, param in netm.named_parameters():
                    new_param = param.detach() - 0.01 * param.grad.detach()
                    param.data = new_param.detach()

                # if i == 0:
                #     print("iter %d ....................." % i)
                #     for name, param in netm.named_parameters():
                #         if param.requires_grad is False:
                #             continue
                        
                #         if not "layers.10" in name:
                #             continue
                        
                #         try:
                #             print("MY W", name, param[:5, :5])
                #         except:
                #             print("MY W", name, param[:5])
                #     print("......................................")
                #     return

                if i == 1:
                    return
                
                if loss.mean() < best_perm_loss:
                    best_perm_loss = loss.mean()
                    best_perm = copy.deepcopy([perm.cpu() for perm in perms])
                    # print("BEST: ", best_perm_loss)

                loss_acum += loss.mean()
                total += 1

            # scheduler.step()

            # total_loss = loss_acum / total
            # if total_loss < best_perm_loss:
            #     best_perm_loss = total_loss
            #     best_perm = copy.deepcopy([perm.cpu() for perm in perms])
            #     print("BEST: ", best_perm_loss)

            print("LOSS: %d" % iteration, loss_acum / total)
            for perm in best_perm:
                print(perm[:11])
        
        return best_perm
    
    def _wrap_network(self, layer_indices, net0, net1, perms):
        wrapped_model = copy.deepcopy(net1)

        layers = list()
        for i, layer_idx in enumerate(layer_indices):
            layer0 = net0.layers[layer_idx]
            layer1 = net1.layers[layer_idx]
            wrapper = ModLinearStraightThroughEstimator

            lst = [
                    wrapper(layer0, layer1, i, perms), 
                    torch.nn.ReLU()
                ]
            if i == 5:
                [
                    wrapper(layer0, layer1, i, perms), 
                    torch.nn.ReLU()
                ]
            layers.extend(
                lst
            )

        wrapped_model.layers = torch.nn.Sequential(*layers)

        return wrapped_model
    
    def _reset_network(self, layer_indices, netm, net1, perms, zero_grad=True):
        for i, layer_idx in enumerate(layer_indices):
            layer1 = net1.layers[layer_idx]

            netm.layers[layer_idx].reset(layer1, perms)
            if zero_grad is True:
                netm.layers.zero_grad()