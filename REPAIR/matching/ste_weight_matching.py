import copy
import tqdm
import torch

from .weight_matching import WeightMatching
from .matching_utils import apply_permutation


class ConvSTEFunc(torch.autograd.Function):
    @staticmethod
    def forward(input, w_for, b_for, w_hat, b_hat, w_back):
        # TODO Implement
        pass

    @staticmethod
    def setup_context(ctx, inputs, output):
        # TODO Implement
        pass

    @staticmethod
    def backward(ctx, grad_output):
        # TODO Implement
        pass


class LinearSTEFunc(torch.autograd.Function):
    @staticmethod
    def forward(input, w_for, b_for, w_hat, b_hat, w_back):
        a = input.mm(w_for.detach().t()) + b_for.detach()
        
        return a 

    @staticmethod
    def setup_context(ctx, inputs, output):
        input, w_for, b_for, w_hat, b_hat, w_back = inputs
        ctx.save_for_backward(input, w_for, b_for, w_hat, b_hat, w_back)

    @staticmethod
    def backward(ctx, grad_output):
        input, _, _, _, _, w_back = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        
        grad_bias   = (0.5 * grad_output).sum(0).detach()
        grad_input  = torch.matmul(grad_output, w_back.detach())
        grad_weight = torch.matmul(0.5 * grad_output.t(), input.detach()) 

        return grad_input, None, None, grad_weight, grad_bias, None


class STEAutograd(torch.nn.Module):
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

        self.w_1 = self.w_1.clone()[p_out, :]
        self.w_1 = self.w_1.clone()[:, p_in]
        self.b_1 = self.b_1.clone()[p_out]

        x = LinearSTEFunc.apply(
            x, 
            0.5 * (self.w_0.detach() + self.w_1.detach()),
            0.5 * (self.b_0.detach() + self.b_1.detach()),
            self.weight,
            self.bias,
            0.5 * (self.w_0.detach() + self.w_1.detach()),
        )
        
        return x
    

class SteMatching:
    def __init__(self, loss_fn, loader, lr, wm, debug=False, epochs=6, ret_perms=False, device="mps"):
        self.debug           = debug
        self.loader          = loader
        self.epochs          = epochs
        self.device          = device
        self.loss_fn         = loss_fn
        self.ret_perms       = ret_perms
        self.lr              = lr
        self.weight_matching = wm
        self.netm            = None

    def __call__(self, layer_indices, net0, net1, debug_perms=None):
        best_perm_loss = 1000.0
        best_perm      = None
        perms          = None

        self.device = "cpu"
        net0.to(self.device)
        net1.to(self.device)

        netm = copy.deepcopy(net0)
        netm = self._wrap_network(layer_indices, net0.to(self.device), net1.to(self.device), copy.deepcopy(perms))

        for iteration in range(self.epochs):
            loss_acum = 0.0
            total = 0

            for i, (images, labels) in enumerate(tqdm.tqdm(self.loader)):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                perms = self.weight_matching(layer_indices, copy.deepcopy(netm), copy.deepcopy(net1), init_perm=perms)
                self._reset_network(layer_indices, netm, net1.to(self.device), copy.deepcopy(perms), zero_grad=True)

                outputs = netm(images)
                loss = torch.nn.functional.cross_entropy(outputs, labels)
                gradient = torch.autograd.grad(loss, netm.parameters())

                for t, grad in zip(netm.named_parameters(), gradient):
                    name = t[0]
                    param = t[1]

                    new_param = param.detach() - self.lr * grad.detach()
                    param.data = new_param.detach()

                if loss.mean() < best_perm_loss:
                    best_perm_loss = loss.mean()
                    best_perm = copy.deepcopy([perm.cpu() for perm in perms])

                loss_acum += loss.mean()
                total += 1

            print("LOSS: %d" % iteration, loss_acum / total)

        if self.ret_perms is True:
            return best_perm
        
        net1 = apply_permutation(layer_indices, net1, best_perm)
        
        return net0, net1
    
    def _wrap_network(self, layer_indices, net0, net1, perms):
        wrapped_model = copy.deepcopy(net1.cpu())

        layers = list()
        for i, layer_idx in enumerate(layer_indices):
            layer0 = net0.layers[layer_idx]
            layer1 = net1.layers[layer_idx]
            wrapper = STEAutograd

            lst = [
                    wrapper(layer0, layer1, i, perms), 
                    torch.nn.ReLU()
                ]
            if i == len(layer_indices) -1:
                lst = [
                    wrapper(layer0, layer1, i, perms), 
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
                netm.layers.zero_grad(set_to_none=True)
