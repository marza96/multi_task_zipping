import copy
import tqdm
import torch
import torchviz

from .matching_utils import apply_permutation
from torch.nn.utils.stateless import functional_call


class Conv2dSTEFunc(torch.autograd.Function):
    @staticmethod
    def forward(input, w_for, b_for, w_hat, b_hat, w_back):
        return torch.nn.functional.conv2d(
            input, 
            w_for.detach(), 
            b_for.detach(), 
            1, 1, 1, 1)

    @staticmethod
    def setup_context(ctx, inputs, output):
        input, w_for, b_for, w_hat, b_hat, w_back = inputs
        ctx.save_for_backward(input, w_for, b_for, w_hat, b_hat, w_back)

    @staticmethod
    def backward(ctx, grad_output):
        input, _, _, _, _, w_back = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        grad_bias = (0.5 * grad_output).sum((0, 2, 3)).squeeze(0)
        grad_input = torch.nn.grad.conv2d_input(input.shape, w_back, grad_output, padding=1) 
        grad_weight = torch.nn.grad.conv2d_weight(input, w_back.shape, 0.5 * grad_output, padding=1)

        return grad_input, None, None, grad_weight, grad_bias, None


class DenseSTEFunc(torch.autograd.Function):
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
    

class STELinear(torch.nn.Module):
    def __init__(self, layer_0, layer_1, p_in, p_out):
        super().__init__()
        
        self.p_in    = p_in
        self.p_out   = p_out
        self.l_type  = type(layer_0)
        self.w_0     = layer_0.weight.detach().float()
        self.w_1     = layer_1.weight.detach().float()
        self.b_0     = layer_0.bias.detach().float()
        self.b_1     = layer_1.bias.detach().float()

        self.weight = torch.nn.Parameter(layer_0.weight.detach().float())
        self.bias   = torch.nn.Parameter(layer_0.bias.detach().float())

        self.weight.requires_grad = True
        self.bias.requires_grad   = True

    def reset(self, layer_1, p_in, p_out):
        self.weight.requires_grad = True
        self.bias.requires_grad   = True

        self.w_1     = layer_1.weight.detach().float()
        self.b_1     = layer_1.bias.detach().float()
        self.p_in    = p_in
        self.p_out   = p_out

    def forward(self, input): 
        x        = input
        self.w_1 = self.w_1.clone()[self.p_out, :]
        self.b_1 = self.b_1.clone()[self.p_out]
        self.w_1 = self.w_1.clone()[:, self.p_in]

        if self.l_type is torch.nn.modules.linear.Linear:
            x = DenseSTEFunc.apply(
                x, 
                0.5 * (self.w_0.detach() + self.w_1.detach()),
                0.5 * (self.b_0.detach() + self.b_1.detach()),
                self.weight,
                self.bias,
                0.5 * (self.w_0.detach() + self.w_1.detach()),
            )
        if self.l_type is torch.nn.modules.conv.Conv2d:
            x = Conv2dSTEFunc.apply(
                x, 
                0.5 * (self.w_0.detach() + self.w_1.detach()),
                0.5 * (self.b_0.detach() + self.b_1.detach()),
                self.weight,
                self.bias,
                0.5 * (self.w_0.detach() + self.w_1.detach()),
            )
        
        return x
    

class STEBnorm(torch.nn.BatchNorm2d):
    def __init__(self, layer_0, layer_1, p_out):
        super().__init__(layer_0.weight.shape[0])

        self.features = layer_0.weight.shape[0]
        device        = layer_1.weight.device
        layer_1.to(device)

        self.weight = torch.nn.Parameter(layer_0.weight.detach().float())
        self.bias   = torch.nn.Parameter(layer_0.bias.detach().float())

        self.register_buffer("running_mean", layer_0.running_mean.detach().float())
        self.register_buffer("running_var", layer_0.running_var.detach().float())
        self.register_buffer("num_batches_tracked", layer_0.num_batches_tracked)

        self.weight.requires_grad              = True
        self.bias.requires_grad                = True

        self.bn_state_dict_1     = layer_1.state_dict()
        self.bn_state_dict_0     = layer_0.state_dict()
        self.bn_state_dict       = {key: None for key in layer_0.state_dict().keys()}

        self._init_bn(p_out)

    def to(self, device):
        for key in self.bn_state_dict_1.keys():
            self.bn_state_dict_1[key].to(device)
            self.bn_state_dict_0[key].to(device)

    def _init_bn(self, p_out):
        for key in self.bn_state_dict.keys():
            if "num_batches_tracked" not in key:
                self.bn_state_dict_1[key] = self.bn_state_dict_1[key].detach()[p_out]
            else:
                self.bn_state_dict_1[key] = self.bn_state_dict_1[key].detach()

            self.bn_state_dict[key]   = self.bn_state_dict_1[key].detach() + (self.state_dict()[key] - self.state_dict()[key].detach())
            self.bn_state_dict[key]   = 0.5 * (self.bn_state_dict_0[key].detach() + self.bn_state_dict[key])

    def reset(self, layer_1, p_in, p_out):
        self.bn_state_dict_1 = layer_1.state_dict()
        self.p_out           = p_out

        self._init_bn(p_out)

    def forward(self, input): 
        x = functional_call(torch.nn.BatchNorm2d(self.features), self.bn_state_dict, input)

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

    def __call__(self, spec, net0, net1):
        best_perm_loss = 1000.0
        best_perm      = None
        perms          = None

        net0.to(self.device)
        net1.to(self.device)

        netm = copy.deepcopy(net0)
        netm = self._wrap_network(spec, net0.to(self.device), net1.to(self.device), copy.deepcopy(perms))

        cnt = 0
        for iteration in range(self.epochs):
            loss_acum = 0.0
            total = 0

            for i, (images, labels) in enumerate(tqdm.tqdm(self.loader)):
                images = images.to(self.device)
                labels = labels.to(self.device)
                cnt += 1
                
                perms = self.weight_matching(spec, copy.deepcopy(netm), copy.deepcopy(net1), init_perm=perms)
                self._reset_network(spec, netm, net1.to(self.device), copy.deepcopy(perms), zero_grad=True)

                outputs = netm(images)
                loss = self.loss_fn(outputs, labels)
                gradient = torch.autograd.grad(loss, netm.parameters())

                if self.debug is True:
                    if cnt == 1: 
                        print("iter %d ....................." % i)
                        for t, grad in zip(netm.named_parameters(), gradient):
                            name = t[0]
                            param = t[1]

                            if not "layers.18" in name:
                                continue
                            
                            try:
                                print("MY G", name, grad[:5, :5])
                            except:
                                print("MY G", name, grad[:5])
                        print("......................................")

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

                if self.debug is True:
                    if cnt == 1:
                        for p in perms:
                            print(p[:11])

                        return

            print("LOSS: %d" % iteration, loss_acum / total)

        if self.ret_perms is True:
            return best_perm
        
        net1 = apply_permutation(spec, net1, best_perm)
        
        return net0, net1
    
    def _wrap_network(self, spec, net0, net1, perms):
        wrapped_model        = copy.deepcopy(net1.cpu())
        perm_spec            = copy.deepcopy(spec.perm_spec)
        layer_spec_unique    = copy.deepcopy(spec.layer_spec_unique)

        net1.to(self.device)
        for i, block in enumerate(layer_spec_unique):
            for j, module in enumerate(block):
                layer_idx = int(module.split(".")[1])

                layer0 = net0.layers[layer_idx]
                layer1 = net1.layers[layer_idx]

                p_in = None
                if len(layer0.weight.shape) > 1:
                    p_in  = torch.arange(layer0.weight.shape[1])

                p_out = torch.arange(layer0.weight.shape[0])

                if perm_spec[i][j][0] != -1:
                    if perms is not None:
                        p_out = perms[perm_spec[i][j][0]]
                
                if perm_spec[i][j][1] != -1:
                    if perms is not None:
                        p_in = perms[perm_spec[i][j][1]]

                if isinstance(layer0, torch.nn.BatchNorm1d) or isinstance(layer0, torch.nn.BatchNorm2d):
                    wrapped_model.layers[layer_idx] = STEBnorm(layer0, layer1, p_out)

                    continue
                
                wrapped_model.layers[layer_idx] = STELinear(layer0, layer1, p_in, p_out)

        return wrapped_model.to(self.device)
    
    def _reset_network(self, spec, netm, net1, perms, zero_grad=True):
        perm_spec            = copy.deepcopy(spec.perm_spec)
        layer_spec_unique    = copy.deepcopy(spec.layer_spec_unique)

        for i, block in enumerate(layer_spec_unique):
            for j, module in enumerate(block):
                layer_idx = int(module.split(".")[1])

                layer1 = net1.layers[layer_idx]

                p_in = None
                if len(layer1.weight.shape) > 1:
                    p_in  = torch.arange(layer1.weight.shape[1])

                p_out = torch.arange(layer1.weight.shape[0])

                if perm_spec[i][j][0] != -1:
                    p_out = perms[perm_spec[i][j][0]]
                
                if perm_spec[i][j][1] != -1:
                    p_in = perms[perm_spec[i][j][1]]

                netm.layers[layer_idx].reset(layer1, p_in, p_out)
                if zero_grad is True:
                    netm.layers.zero_grad(set_to_none=True)
