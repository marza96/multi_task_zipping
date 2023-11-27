import copy
import tqdm
import torch
import torch.nn.functional as F

from torch import Tensor
from torch.nn.modules.module import Module
from torch.optim import SGD

from .weight_matching import WeightMatching


class LinearSTEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, p_in, p_out, w_0, b_0, w_1, b_1, w_hat):
        ctx.save_for_backward(w_0, w_hat)

        w_1  = w_1[p_out, :]
        w_1  = w_1[:, p_in]

        a    = F.linear(input.float(), w_0, b_0)
        b_pi = F.linear(input.float(), w_1 , b_1[p_out])

        return 0.5 * (a + b_pi)

    @staticmethod
    def backward(ctx, grad_output):
        w_0, w_hat, = ctx.saved_tensors

        return grad_output @ (0.5 * (w_0 + w_hat)), None, None, None, None, None, None, None
    

class LinearStraightThroughEstimator(torch.nn.Module):
    def __init__(self, layer_0, layer_1, idx, device="mps"):
        super().__init__()
        
        self.perms     = None
        self.idx       = idx
        self.layer_0   = layer_0.to(device)
        self.layer_1   = layer_1.to(device)
        self.layer_hat = copy.deepcopy(layer_0).to(device)

    def set_perms(self, perms):
        self.perms = perms

    def forward(self, input):
        p_in  = None
        p_out = None

        x= input

        if self.idx == 0:
            p_in = torch.arange(x.shape[1]).long()
            p_in.to(x.device)
        elif self.idx > 0:
            p_in = self.perms[self.idx - 1]
            p_in.to(x.device)

        if self.idx < len(self.perms) - 1:
            p_out = self.perms[self.idx]
            p_out.to(x.device)
        else:
            p_out = torch.arange(self.layer_hat.weight.shape[0])
            p_out.to(x.device)
        
        x = LinearSTEFunction.apply(
            x, 
            p_in.to(x.device), 
            p_out.to(x.device),
            self.layer_0.weight,
            self.layer_0.bias,
            self.layer_1.weight,
            self.layer_1.bias,
            self.layer_hat.weight
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

        self.net0 = None
        self.net1 = None
        self.netm = None

    def __call__(self, layer_indices, net0, net1):
        if self.net0 is None and self.net1 is None:
            self.net0 = net0
            self.net1 = net1
            self.netm = self._wrap_network(
                layer_indices, 
                net0, 
                net1
            )

            self.net0.to(self.device)
            self.net1.to(self.device)
            self.netm.to(self.device)

            for param in self.net0.parameters():
                param.requires_grad = False
            for param in self.net1.parameters():
                param.requires_grad = False

        self.netm.train()

        optimizer = SGD(self.netm.parameters(), lr=0.1, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.75)

        best_perm_loss = 1000.0
        best_perm = None

        for iteration in range(self.epochs):
            loss_acum = 0.0
            total = 0

            perms = self.weight_matching(layer_indices, self.netm, self.net1, ste=True)

            for i, (images, labels) in enumerate(tqdm.tqdm(self.loader)):

                optimizer.zero_grad(set_to_none=True)

                # perms = self.weight_matching(layer_indices, self.netm, self.net1, ste=True)

                for i, layer_idx in enumerate(layer_indices):
                    self.netm.layers[layer_idx].set_perms(perms)

                outputs = self.netm(images.to(self.device))
                # loss    = self.loss_fn(outputs, labels.to(self.device))

                loss    = torch.nn.functional.nll_loss(outputs, labels.to(self.device))

                loss.backward()
                optimizer.step()

                # if loss.mean() < best_perm_loss:
                #     best_perm_loss = loss.mean()
                #     best_perm = copy.deepcopy([perm.cpu() for perm in perms])
                #     # print("BEST: ", best_perm_loss)

                loss_acum += loss.mean()
                total += 1

            # scheduler.step()

            total_loss = loss_acum / total
            if total_loss < best_perm_loss:
                best_perm_loss = total_loss
                best_perm = copy.deepcopy([perm.cpu() for perm in perms])
                print("BEST: ", best_perm_loss)

            print("LOSS: %d" % iteration, loss_acum / total)
            for perm in best_perm:
                print(perm[:11])
        
        return best_perm

    def _wrap_network(self, layer_indices, net0, net1):
        wrapped_model = copy.deepcopy(net1)

        for i, layer_idx in enumerate(layer_indices):
            layer0 = net0.layers[layer_idx]
            layer1 = net1.layers[layer_idx]

            wrapper = LinearStraightThroughEstimator

            wrapped_model.layers[layer_idx] = wrapper(layer0, layer1, i)

        return wrapped_model
