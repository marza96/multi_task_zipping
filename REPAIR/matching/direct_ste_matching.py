import torch
import copy
import tqdm
import wandb

from collections import defaultdict
from .matching_utils import perm_to_permmat, permmat_to_perm

def auction_lap(X, eps, compute_score=False):
    cost     = torch.zeros((1, X.shape[1])).to("cuda")
    curr_ass = (torch.zeros(X.shape[0]).long() - 1).to("cuda")
    bids     = torch.zeros(X.shape).to("cuda")
    
    counter = 0
    while (curr_ass == -1).any():
        counter += 1
        unassigned = (curr_ass == -1).nonzero().squeeze()
        
        value = X[unassigned] - cost
        top_value, top_idx = value.topk(2, dim=1)
        
        first_idx = top_idx[:,0]
        first_value, second_value = top_value[:,0], top_value[:,1]
        
        bid_increments = first_value - second_value + eps
        
        bids_ = bids[unassigned]
        bids_.zero_()
        if unassigned.dim()==0:
            high_bids, high_bidders = bid_increments, unassigned
            cost[:,first_idx] += high_bids
            curr_ass[(curr_ass == first_idx).nonzero()] = -1
            curr_ass[high_bidders] = first_idx
        else:
            bids_.scatter_(
                dim=1,
                index=first_idx.contiguous().view(-1, 1),
                src=bid_increments.view(-1, 1)
            )
        
            have_bidder = (bids_ > 0).int().sum(dim=0).nonzero()
            
            high_bids, high_bidders = bids_[:,have_bidder].max(dim=0)
            high_bidders = unassigned[high_bidders.squeeze()]
            
            cost[:,have_bidder] += high_bids
            
            ind = (curr_ass.view(-1, 1) == have_bidder.view(1, -1)).sum(dim=1).nonzero()
            curr_ass[ind] = -1
            
            curr_ass[high_bidders] = have_bidder.squeeze()
        
    score = None
    if compute_score:
        score = int(X.gather(dim=1, index=curr_ass.view(-1, 1)).sum())
    
    return score, perm_to_permmat(curr_ass), counter


class LayerWrapper(torch.nn.Module):
    def __init__(self, layer0, layer1, alpha, device):
        super().__init__()
        self.is_bnorm  = False
        self.alpha     = alpha
        self.device    = device
        self.perm_dict = None
        self.layer_idx = None

        self.layer = copy.deepcopy(layer0).to(self.device)

        self.sd0 = {key: layer0.state_dict()[key].detach().clone() for key in layer0.state_dict()}
        self.sd1 = {key: layer1.state_dict()[key].detach().clone() for key in layer1.state_dict()}
        self.sd  = {key: layer1.state_dict()[key].detach().clone() for key in layer1.state_dict()}

        for key in self.sd0.keys():
            self.sd0[key].requires_grad = False
            self.sd1[key].requires_grad = False
            self.sd[key].requires_grad  = False

    def forward(self, x):
        for param_name in self.layer.state_dict().keys():
            if 'num_batches_tracked' in param_name:
                continue
            
            self.sd[param_name] = self.sd1[param_name].detach().clone()

            for axis in self.perm_dict[param_name]:
                perm_idx = self.perm_dict[param_name][axis]["perm_idx"]
                if perm_idx == -1:
                    continue
                
                if axis == 0:
                    lp = self.latent_perms[perm_idx]
                    p  = self.perms[perm_idx]

                    self.sd[param_name] = (p.detach() + lp - lp.detach()) @ self.sd[param_name]

                    continue
                
                lp = self.latent_perms[perm_idx]
                p  = self.perms[perm_idx]

                self.sd[param_name] = self.sd[param_name] @ (p.detach() + lp - lp.detach()).T

            self.sd[param_name] = 0.5 * (self.sd[param_name] + self.sd0[param_name].detach())          
            
            if "running_" in param_name or "batches" in param_name:
                self.layer.__dict__[param_name] = self.sd[param_name].detach()

                continue

            self.layer.__dict__[param_name] = self.sd[param_name]
        
        return self.layer(x)

    def set_perms(self, perm_dict, perms, latent_perms, layer_idx):
        self.perm_dict    = perm_dict
        self.perms        = perms
        self.latent_perms = latent_perms
        self.layer_idx    = layer_idx


class DirectSTE:
    def __init__(self, loss_fn, loader, lr, debug=False, epochs=6, ret_perms=False, device="mps", exp_name="dummy"):
        self.debug           = debug
        self.loader          = loader
        self.epochs          = epochs
        self.device          = device
        self.loss_fn         = loss_fn
        self.ret_perms       = ret_perms
        self.lr              = lr
        self.exp_name        = exp_name

    @staticmethod
    def set_subdict(perm_dict, perms, latent_perms, perm_axes, except_axis=None, key=None):
        layer_idx = key.split(".")[1]
        param_name = key.split(".")[-1]
        for ax_idx, p in enumerate(perm_axes):
            if ax_idx == except_axis:
                continue
            
            if layer_idx not in perm_dict.keys():
                perm_dict.update({layer_idx: {}})

            if param_name not in perm_dict[layer_idx].keys():
                perm_dict[layer_idx].update({param_name: {}})
            
            if ax_idx not in perm_dict[layer_idx][param_name].keys():
                perm_dict[layer_idx][param_name].update({ax_idx: {}})

            perm_dict[layer_idx][param_name][ax_idx] = {"perm_idx": p}

    def __call__(self, spec, net0, net1):
        best_perm_loss = 1000.0
        best_perm      = None

        net0.to(self.device)
        net1.to(self.device)

        layer_specs = spec.layer_spec
        perm_specs  = spec.perm_spec
        perm_mats   = [None for _ in range(len(perm_specs))]
        latent_perm_mats   = [None for _ in range(len(perm_specs))]
        perms              = [None for _ in range(len(perm_specs))]
        perm_dict          = defaultdict()

        for i in range(len(perm_specs)):
            layer_idx           = int(layer_specs[i][0].split(".")[1])
            w_shape             = net0.layers[layer_idx].weight.shape
            perm_mats[i]        = torch.eye(w_shape[0]).to(self.device)
            latent_perm         = -1.0 * torch.rand(perm_mats[i].shape).to(self.device)
            latent_perm_mats[i] = torch.nn.Parameter(latent_perm)
            perms[i]            = torch.arange(w_shape[0]).to(self.device)

        wrapped_model = self._wrap_network(spec, net0, net1)
        optim = torch.optim.SGD(latent_perm_mats, self.lr, momentum=0.85)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, 50)

        self.set_perm_dict(perm_dict, spec, net0, perm_mats, latent_perm_mats)
    
        cnt = 0
        for iteration in range(self.epochs):
            loss_acum = 0.0
            total = 0

            for i, (images, labels) in enumerate(tqdm.tqdm(self.loader)):
                optim.zero_grad(set_to_none=True)

                images = images.to(self.device)
                labels = labels.to(self.device)
                cnt += 1

                self.solve_laps(perm_mats, latent_perm_mats, cnt)
                self.adjust_perms(spec, wrapped_model, perm_dict, perm_mats, latent_perm_mats)

                outputs = wrapped_model(images)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                optim.step()
                # scheduler.step()

                wandb.log({"STE loss": loss.mean()})

                loss_acum += loss.mean()
                total += 1
                cnt += 1

            print("LOSS: %d" % iteration, loss_acum / total) 

        final_perms = [permmat_to_perm(m) for m in perm_mats]

        if self.ret_perms is True:
            return final_perms

        net1 = self.apply_permutation(spec, net1, final_perms)

        return net0.to(self.device), net1.to(self.device)

    def log_sinkhorn(self, log_alpha, n_iter):
        for _ in range(n_iter):
            log_alpha = log_alpha - torch.logsumexp(log_alpha, -1, keepdim=True)
            log_alpha = log_alpha - torch.logsumexp(log_alpha, -2, keepdim=True)
        return log_alpha.exp()

    def solve_laps(self, perm_mats, latent_perm_mats, iter):
        for i in range(len(perm_mats)):
            increment = 0.5 * (torch.min(torch.abs(latent_perm_mats[i])) + 1.0)

            obj = latent_perm_mats[i] - torch.log(-torch.log(torch.rand(latent_perm_mats[i].shape))).to(self.device)
            obj = self.log_sinkhorn(obj, 10)

            perm_mats[i] = auction_lap(obj, 1e-2)[1].to(self.device)
    
    def adjust_perms(self, spec, net, perm_dict, perms, latent_perms):
        layer_spec_unique    = copy.deepcopy(spec.layer_spec_unique)

        for i, block in enumerate(layer_spec_unique):
            for j, module in enumerate(block):
                layer_idx = int(module.split(".")[1])

                try:
                    net.layers[layer_idx].set_perms(perm_dict[str(layer_idx)], perms, latent_perms, layer_idx)
                except:
                    pass
        
    def _wrap_network(self, spec, net0, net1):
        wrapped_model        = copy.deepcopy(net1).to(self.device)
        layer_spec_unique    = copy.deepcopy(spec.layer_spec_unique)

        for i, block in enumerate(layer_spec_unique):
            for j, module in enumerate(block):
                layer_idx = int(module.split(".")[1])

                layer0 = net0.layers[layer_idx]
                layer1 = net1.layers[layer_idx]

                wrapped_model.layers[layer_idx] = LayerWrapper(layer0, layer1, 1000.0, self.device)

        return wrapped_model.to(self.device)
    
    def set_perm_dict(self, perm_dict, spec, net, perm_mats, latent_perm_mats):
        for i in range(len(spec.perm_spec)):
            perm_spec  = copy.deepcopy(spec.perm_spec[i])
            layer_spec = copy.deepcopy(spec.layer_spec[i])

            assert len(perm_spec) == len(layer_spec)

            for j in range(len(perm_spec)):
                layer_key = layer_spec[j]
                perm_axes = perm_spec[j]

                self.set_subdict(
                    perm_dict, perm_mats, latent_perm_mats, perm_axes, except_axis=None, key=layer_key
                )

    @staticmethod
    def get_permuted_param(param, perms, perm_axes, except_axis=None):
        w = param

        for ax_idx, p in enumerate(perm_axes):
            if ax_idx == except_axis:
                continue
            
            if p != -1:
                w = torch.index_select(w, ax_idx, perms[p].int().cpu())

        return w
    
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