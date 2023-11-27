import copy
import tqdm
import torch
import torch.nn.functional as F

from torch import Tensor
from torch.nn.modules.module import Module
from torch.optim import SGD

from .weight_matching import WeightMatching

from torch.nn.utils.stateless import functional_call

from torchviz import make_dot


class LinearSTEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, p_in, p_out, w_0, b_0, w_1, b_1, w_hat, b_hat, w_hat_d, idx):
        ctx.save_for_backward(input, w_0, w_hat_d, idx)

        w_1  = w_1[p_out, :]
        w_1  = w_1[:, p_in]

        a    = F.linear(input, 0.5 * w_0, 0.5 * b_0)
        b_pi = F.linear(input, 0.5 * w_1 , 0.5 * b_1[p_out])

        return (a + b_pi).detach()

    @staticmethod
    def backward(ctx, grad_output):
        input, w_0, w_hat, idx = ctx.saved_tensors

        # print("input_grad:", idx, " ", grad_output[0, :19])

        grad_input  = grad_output.mm(0.5 * (w_0 + w_hat))
        grad_weight = grad_output.t().mm(input) 
        grad_bias   = grad_output.sum(0)

        return grad_input, None, None, None, None, None, None, grad_weight, grad_bias, None, None
    

class LinearStraightThroughEstimator(torch.nn.Module):
    def __init__(self, layer_0, layer_1, idx, device="mps"):
        super().__init__()
        
        self.perms     = None
        self.idx       = idx

        self.w_hat = layer_0.weight.clone()

        self.w_0 = layer_0.weight.clone()
        self.w_1 = layer_1.weight.clone()
        self.b_0 = layer_0.bias.clone()
        self.b_1 = layer_1.bias.clone()

        self.weight = torch.nn.Parameter(layer_0.weight.clone())
        self.bias = torch.nn.Parameter(layer_0.bias.clone())

        self.weight.requires_grad = True
        self.bias.requires_grad = True

    def set_perms(self, perms):
        self.perms = perms

    def forward(self, input):
        p_in  = None
        p_out = None

        x = input

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
        
        x = LinearSTEFunction.apply(
            x, 
            p_in.to(x.device), 
            p_out.to(x.device),
            self.w_0,
            self.b_0,
            self.w_1,
            self.b_1,
            self.weight,
            self.bias,
            self.w_hat,
            torch.Tensor([self.idx, ])
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

    def __call__(self, layer_indices, net0, net1, debug_perms=None):
        if self.net0 is None and self.net1 is None:
            self.net0 = net0
            self.net1 = net1

            for name, param in self.net0.named_parameters():
                param.requires_grad = False
            for name, param in self.net1.named_parameters():
                param.requires_grad = False

            self.netm = self._wrap_network(
                layer_indices, 
                net0, 
                net1
            )

            self.net0.to(self.device)
            self.net1.to(self.device)
            self.netm.to(self.device)

            for name, param in self.netm.named_parameters():
                if "hat" in name:
                    param.requires_grad = True

        # optimizer = SGD(self.netm.parameters(), lr=0.05, momentum=0.9)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.75)

        best_perm_loss = 1000.0
        best_perm = None
        perms = None

        loss_fn = torch.nn.NLLLoss()
        with torch.autograd.enable_grad():
            for iteration in range(self.epochs):
                loss_acum = 0.0
                total = 0

                perms = self.weight_matching(layer_indices, self.netm, self.net1, init_perm=perms)

                for i, (images, labels) in enumerate(tqdm.tqdm(self.loader)):
                    # optimizer.zero_grad()
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    self.netm.zero_grad()

                    for i, layer_idx in enumerate(layer_indices):
                        self.netm.layers[layer_idx].set_perms(copy.deepcopy(perms))

                    outputs = self.netm(images)

                    print("OUT MY", outputs[0, :11])
                
                    loss = loss_fn(outputs, labels)
                    loss.backward()

                    # print("MY LOSS", loss.item())
                    # print("MY LOSS GRAD", loss.grad)

                    # make_dot(outputs, params=dict(list(self.netm.named_parameters()))).render("rnn_torchviz", format="png")

                
                    # for perm in perms:
                    #     print(perm[:11])
                        
                    print("---------------")
                    for name, param in self.netm.named_parameters():
                        if param.requires_grad is False:
                            continue
                        
                        print("PARAM", name)
                        try:
                            print(name, param.grad[:5, :5])
                        except:
                            print(name, param.grad[:5])
                    print("......................................")
                    return
                    
                    # optimizer.step()

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

        layers = list()
        for i, layer_idx in enumerate(layer_indices):
            layer0 = net0.layers[layer_idx]
            layer1 = net1.layers[layer_idx]
            wrapper = LinearStraightThroughEstimator

            layers.extend(
                [
                    wrapper(layer0, layer1, i), 
                    torch.nn.ReLU()
                ]
            )

        wrapped_model.layers = torch.nn.Sequential(*layers)

        return wrapped_model
