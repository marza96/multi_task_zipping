import copy
import tqdm
import torch
import torchviz
import wandb

from torch.nn.utils.stateless import functional_call


class Conv2dSTEFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, w_for, b_for, w_hat, b_hat, w_back):
        ctx.save_for_backward(input, w_for, b_for, w_hat, b_hat, w_back)

        return torch.nn.functional.conv2d(
            input, 
            w_for.detach(), 
            b_for.detach(), 
            1, 1, 1, 1)

    # @staticmethod
    # def setup_context(ctx, inputs, output):
    #     input, w_for, b_for, w_hat, b_hat, w_back = inputs
    #     ctx.save_for_backward(input, w_for, b_for, w_hat, b_hat, w_back)

    @staticmethod
    def backward(ctx, grad_output):
        input, _, _, _, _, w_back, = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        grad_bias = (0.5 * grad_output).sum((0, 2, 3)).squeeze(0)
        grad_input = torch.nn.grad.conv2d_input(input.shape, w_back, grad_output, padding=1) 
        grad_weight = torch.nn.grad.conv2d_weight(input, w_back.shape, 0.5 * grad_output, padding=1)

        return grad_input, None, None, grad_weight, grad_bias, None


# TODO Fix autograd function format depending on pytorch version
    
class DenseSTEFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, w_for, b_for, w_hat, b_hat, w_back):
        ctx.save_for_backward(input, w_for, b_for, w_hat, b_hat, w_back)

        a = input.mm(w_for.detach().t()) + b_for.detach()
        
        return a 

    # @staticmethod
    # def setup_context(ctx, inputs, output):
    #     input, w_for, b_for, w_hat, b_hat, w_back = inputs
    #     ctx.save_for_backward(input, w_for, b_for, w_hat, b_hat, w_back)

    @staticmethod
    def backward(ctx, grad_output):
        input, _, _, _, _, w_back, = ctx.saved_tensors
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
    

class STEBnorm(torch.nn.Module):
    def __init__(self, layer_0, layer_1, p_out, conv=True):
        super().__init__()

        self.conv     = conv
        self.features = layer_0.weight.shape[0]
        self.weight   = torch.nn.Parameter(layer_0.weight.detach().float().clone())
        self.bias     = torch.nn.Parameter(layer_0.bias.detach().float().clone())

        self.register_buffer("running_mean", layer_0.running_mean.detach().float().clone())
        self.register_buffer("running_var", layer_0.running_var.detach().float().clone())
        self.register_buffer("num_batches_tracked", layer_0.num_batches_tracked.detach().clone())

        self.weight.requires_grad = True
        self.bias.requires_grad   = True
        
        self.bn_state_dict   = {key: None for key in layer_0.state_dict().keys()}
        self.bn_state_dict_0 = {key: None for key in layer_0.state_dict().keys()}
        self.bn_state_dict_1 = {key: None for key in layer_0.state_dict().keys()}

        for key in self.state_dict():
            self.bn_state_dict_0[key] = torch.empty_like(layer_0.state_dict()[key])
            self.bn_state_dict_1[key] = torch.empty_like(layer_0.state_dict()[key])
            self.bn_state_dict[key]   = torch.empty_like(layer_0.state_dict()[key])

            self.bn_state_dict_0[key] = layer_0.state_dict()[key].clone().detach()
            self.bn_state_dict_1[key] = layer_1.state_dict()[key].clone().detach()

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

            if "weight" in key:
                self.bn_state_dict[key] = self.bn_state_dict_1[key].detach() + (self.weight - self.weight.detach())
                self.bn_state_dict[key] = 0.5 * (self.bn_state_dict_0[key].detach() + self.bn_state_dict[key])
                
                continue

            if "bias" in key:
                self.bn_state_dict[key] = self.bn_state_dict_1[key].detach() + (self.bias - self.bias.detach())
                self.bn_state_dict[key] = 0.5 * (self.bn_state_dict_0[key].detach() + self.bn_state_dict[key])
                
                continue
                
            self.bn_state_dict[key] = self.bn_state_dict_1[key].detach() + (self.state_dict()[key] - self.state_dict()[key].detach())
            self.bn_state_dict[key] = 0.5 * (self.bn_state_dict_0[key].detach() + self.bn_state_dict[key])

    def reset(self, layer_1, p_in, p_out):
        for key in self.state_dict():
            self.bn_state_dict_1[key] = layer_1.state_dict()[key].clone().detach()

        self.p_out = p_out

        self._init_bn(p_out)

    def forward(self, input): 
        if self.conv is True:
            x = functional_call(torch.nn.BatchNorm2d(self.features), self.bn_state_dict, input)

            return x
        
        x = functional_call(torch.nn.BatchNorm1d(self.features), self.bn_state_dict, input)
        
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

    def get_l2(self, net0, net1, best_perm):
        net1_p = self.apply_permutation(
            net1.perm_spec, 
            net1, 
            best_perm
        )

        sd1 = net1_p.state_dict()
        sd0 = net0.state_dict()

        dist = 0.0
        for key in sd0.keys():
            if "num_batches_tracked" in key:
                continue
            dist += torch.norm(sd0[key] - sd1[key])

        return dist

    def __call__(self, spec, net0, net1):
        best_perm_loss = 1000.0
        best_perm      = None
        perms          = None

        net1_cpy = copy.deepcopy(net1.to("cpu")).to(self.device)

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
                
                perms = self.weight_matching(spec, netm, copy.deepcopy(net1), init_perm=perms)
                netm.to(self.device)
                
                self._reset_network(spec, netm, net1.to(self.device), copy.deepcopy(perms), zero_grad=True)

                outputs = netm(images)
                loss = self.loss_fn(outputs, labels)
                gradient = torch.autograd.grad(loss, netm.parameters())

                if best_perm is not None:
                    l2_dist = self.get_l2(net0, net1_cpy, best_perm)
                    wandb.log({"STE loss": loss.mean(), "L2 dist": l2_dist})
                else:
                    wandb.log({"STE loss": loss.mean()})
  
                if self.debug is True:
                    if cnt == 200: 
                        print("iter %d ....................." % i)
                        for t, grad in zip(netm.named_parameters(), gradient):
                            name = t[0]
                            param = t[1]

                            if not "layers.4" in name:
                                continue
                            
                            try:
                                print("MY GEN G", name, grad[:5, :5])
                            except:
                                print("MY GEN G", name, grad[:5])
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
                    if cnt == 200:
                        for p in perms:
                            print(p[:11])

                        return
            
            print("LOSS: %d" % iteration, loss_acum / total)

        if self.ret_perms is True:
            return best_perm
        
        # NOTE Changed in comparison to old code
        net1 = self.apply_permutation(spec, net1, best_perm)
        
        print("MY BEST PERM:")
        for p in best_perm:
            print(p[:11])

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
                    wrapped_model.layers[layer_idx] = STEBnorm(layer0, layer1, p_out, isinstance(layer0, torch.nn.BatchNorm2d))

                    continue
                
                wrapped_model.layers[layer_idx] = STELinear(layer0, layer1, p_in, p_out)

        return wrapped_model.to(self.device)
    
    @staticmethod
    def get_permuted_param(param, perms, perm_axes, except_axis=None):
        w = param

        for ax_idx, p in enumerate(perm_axes):
            if ax_idx == except_axis:
                continue
            
            if p != -1:
                w = torch.index_select(w, ax_idx, perms[p].int())

        return w
    
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

    def apply_permutation(self, spec, net, perms):
        net_state_dict = net.state_dict()

        for i in range(len(spec.perm_spec)):
            perm_spec  = copy.deepcopy(spec.perm_spec[i])
            layer_spec = copy.deepcopy(spec.layer_spec[i])

            assert len(perm_spec) == len(layer_spec)

            for j in range(len(perm_spec)):
                layer_key = layer_spec[j]
                perm_axes = perm_spec[j]

                perm_param = self.get_permuted_param(
                    net_state_dict[layer_key].cpu(),
                    perms,
                    perm_axes,
                    except_axis=None
                ).to(self.device)

                net_state_dict[layer_key] = perm_param

        net.load_state_dict(net_state_dict)

        return net
