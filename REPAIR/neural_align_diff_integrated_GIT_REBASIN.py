import torch
import scipy
import copy
import math

from tqdm import tqdm
from collections import namedtuple, defaultdict, OrderedDict
from typing import NamedTuple

from .net_models.mlp import LayerWrapper, LayerWrapper2D, CNN
from .matching.weight_matching import WeightMatching
from .matching.activation_matching import activation_matching
from .matching.ste_matching import SteMatching
import numpy as np
import torch.nn as nn

import matplotlib.pyplot as plt
from torchviz import make_dot


torch.set_printoptions(precision=5, sci_mode=False)


def flatten_params(model):
    return model.state_dict()

def lerp(lam, t1, t2):
    t3 = copy.deepcopy(t1)
    for p in t1:
        t3[p] = (1 - lam) * t1[p] + lam * t2[p]
    return t3

def freeze(x):
    ret = copy.deepcopy(x)
    for key in x:
        ret[key] = x[key].detach()
    return ret

def clone(x):
    ret = OrderedDict()
    for key in x:
        ret[key] = x[key].clone().detach()
    return ret

def wm_learning(model_a, model_b, train_loader, permutation_spec, dbg_perm=None):
    from torch.nn.utils.stateless import functional_call
    import torchopt

    device = "mps"

    # Best permutation found so far...
    best_perm = None
    best_perm_loss = 999
    perm = None

    train_state = model_a.state_dict()
    model_target = copy.deepcopy(model_a)

    optimizer = torchopt.sgd(lr=0.05, momentum=0.9, moment_requires_grad=True)

    for key in train_state:
        train_state[key] = train_state[key].float()

    opt_state = optimizer.init(train_state)

    for epoch in tqdm(range(0, 20)):
        correct = 0.
        loss_acum = 0.0
        total = 0

        for i, (x, t) in enumerate(tqdm(train_loader, leave=False)):
            x = x.to(device)
            t = t.to(device)

            # projection by weight matching
            perm, _ = weight_matching_ref(permutation_spec,
                                        train_state, flatten_params(model_b),
                                        max_iter=100, print_flg=False, debug_perms=dbg_perm)

            for p in perm:
                print(perm[p][:11])
            projected_params = apply_permutation(permutation_spec, perm, flatten_params(model_b))


            # ste
            for key in train_state:
                train_state[key] = train_state[key].detach()  # leaf
                train_state[key].requires_grad = True
                train_state[key].grad = None  # optimizer.zero_grad()


            # straight-through-estimator https://github.com/samuela/git-re-basin/blob/main/src/mnist_mlp_ste2.py#L178
            ste_params = {}
            for key in projected_params:
                ste_params[key] = projected_params[key].detach() + (train_state[key] - train_state[key].detach())

            midpoint_params = lerp(0.5, freeze(flatten_params(model_a)), ste_params)
            
            model_target.train()


            output = functional_call(model_target, midpoint_params, x)
            print("OUT LEG", output[0, :11])
            # return

            make_dot(output, params=dict(list(model_target.named_parameters()))).render("leg_tv", format="png")
            

            loss = torch.nn.functional.nll_loss(output, t)
            print("LEG LOSS", loss.item())
            
            loss.backward()

            for key in train_state:
                try:
                    print(key, train_state[key].grad[:5, :5])
                except:
                    print(key, train_state[key].grad[:5])

            print("......................................")
            return

            # optimize
            grads = OrderedDict()
            for key in train_state:
                if train_state[key].grad is None:
                    grads[key] = torch.zeros_like(train_state[key])
                else:
                    grads[key] = train_state[key].grad
            for key in train_state:
                train_state[key] = train_state[key].detach()  # avoid opt_sate chain
            updates, opt_state = optimizer.update(grads, opt_state, params=train_state, inplace=False)
            train_state = torchopt.apply_updates(train_state, updates, inplace=False)

            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(t.view_as(pred)).sum().item()

            if loss < best_perm_loss:
                best_perm_loss = loss.item()
                best_perm = perm
            
            loss_acum += loss.mean()
            total += 1

        print("LOSS: %d" % i, loss_acum / total)

    final_perm = [None for _ in range(5)]
    for key in best_perm.keys():
        idx = key.split("_")[1]
        final_perm[int(idx)] = best_perm[key].long()

    return final_perm


class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return torch.nn.functional.F.hardtanh(grad_output)


class StraightThroughEstimator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
            x = STEFunction.apply(x)
            return x

class PermutationSpec(NamedTuple):
    perm_to_axes: dict
    axes_to_perm: dict

def get_permuted_param(ps: PermutationSpec, perm, k: str, params, except_axis=None):
    """Get parameter `k` from `params`, with the permutations applied."""
    w = params[k]

    # if k == 'classifier.weight':  # to reshape because of input shape is 3x 96 x 96
    #     w = w.reshape(126, 512 * 4, 3, 3)
    try:
        for axis, p in enumerate(ps.axes_to_perm[k]):
            # Skip the axis we're trying to permute.
            if axis == except_axis:
                continue

            # None indicates that there is no permutation relevant to that axis.
            if p is not None:
                w = torch.index_select(w, axis, perm[p].int())
        # if k == 'classifier.weight':
        #     w = w.reshape(126, -1)
    except KeyError:
        pass

    return w


def apply_permutation(ps: PermutationSpec, perm, params):
    """Apply a `perm` to `params`."""
    ret = {}
    for k in params.keys():
        if params[k].dim() != 0:  # avoid num_batches_tracked
            ret[k] = get_permuted_param(ps, perm, k, params)
        else:
            ret[k] = params[k]

    return ret


def weight_matching_ref(ps: PermutationSpec, params_a, params_b, max_iter=300, debug_perms=None, init_perm=None, print_flg=True, legacy=True):
    """Find a permutation of `params_b` to make them match `params_a`."""
    perm_sizes = {p: params_a[axes[0][0]].shape[axes[0][1]] for p, axes in ps.perm_to_axes.items()}
    device = list(params_a.values())[0].device
    perm = {p: torch.arange(n) for p, n in perm_sizes.items()} if init_perm is None else init_perm
    perm = {key: perm[key].cpu() for key in perm}  # to cpu
    params_a = {key: params_a[key].cpu() for key in params_a}  # to cpu
    params_b = {key: params_b[key].cpu() for key in params_b}  # to cpu
    perm_names = list(perm.keys())
    metrics = {'step': [], 'l2_dist': []}
    step = 0
    for iteration in range(max_iter):
        progress = False

        rperm = torch.randperm(len(perm_names))
        if debug_perms is not None:
            rperm = debug_perms[iteration]

        for p_ix in rperm:
            p = perm_names[p_ix]
            n = perm_sizes[p]
            A = torch.zeros((n, n))
            for wk, axis in ps.perm_to_axes[p]:  # layer loop
                if ('running_mean' not in wk) and ('running_var' not in wk) and ('num_batches_tracked' not in wk):
                    w_a = params_a[wk]  # target
                    w_b = get_permuted_param(ps, perm, wk, params_b, except_axis=axis)
                    w_a = torch.moveaxis(w_a, axis, 0).reshape((n, -1))
                    w_b = torch.moveaxis(w_b, axis, 0).reshape((n, -1))
                    A += w_a @ w_b.T  # A is cost matrix to assignment,
                    # print(torch.sum(w_a @ w_b.T))

            # print("..............")
            ri, ci = scipy.optimize.linear_sum_assignment(A.detach().numpy(), maximize=True)
            assert (torch.tensor(ri) == torch.arange(len(ri))).all()
            oldL = torch.einsum('ij,ij->i', A, torch.eye(n)[perm[p].long()]).sum()
            newL = torch.einsum('ij,ij->i', A, torch.eye(n)[ci, :]).sum()
            if print_flg:
                print(f"{iteration}/{p}: {newL - oldL}")
            progress = progress or newL > oldL + 1e-12

            perm[p] = torch.Tensor(ci)
            p_params_b = apply_permutation(ps, perm, params_b)
            # l2_dist = get_l2(params_a, p_params_b)
            # metrics['step'].append(step)
            # metrics['l2_dist'].append(l2_dist)
            step += 1
        if not progress:
            break

    perm = {key: perm[key].to("mps") for key in perm}  # to device
    final_perm = [None for _ in range(5)]
    for key in perm.keys():
        idx = key.split("_")[1]
        final_perm[int(idx)] = perm[key].long()
    params_a = {key: params_a[key].to(device) for key in params_a}  # to device
    params_b = {key: params_b[key].to(device) for key in params_b}  # to device

    # IF used in my framework then uncomment this

    if legacy is True:
        return perm, metrics
    
    return final_perm, metrics


def permutation_spec_from_axes_to_perm(axes_to_perm: dict) -> PermutationSpec:
    perm_to_axes = defaultdict(list)
    for wk, axis_perms in axes_to_perm.items():
        for axis, perm in enumerate(axis_perms):
            if perm is not None:
                perm_to_axes[perm].append((wk, axis))
    return PermutationSpec(perm_to_axes=dict(perm_to_axes), axes_to_perm=axes_to_perm)


def mlp_permutation_spec(num_hidden_layers: int, bias_flg: bool) -> PermutationSpec:
    """We assume that one permutation cannot appear in two axes of the same weight array."""
    assert num_hidden_layers >= 1

    if bias_flg:
        print({"layers.0.weight": ("P_0", None),
            **{f"layers.{2*i}.weight": (f"P_{i}", f"P_{i - 1}")
               for i in range(1, num_hidden_layers)},
            **{f"layers.{2*i}.bias": (f"P_{i}",)
               for i in range(num_hidden_layers)},
            f"layer{2 * num_hidden_layers}.weight": (None, f"P_{num_hidden_layers - 1}"),
            f"layer{2 * num_hidden_layers}.bias": (None,)})
        return permutation_spec_from_axes_to_perm({
            "layers.0.weight": ("P_0", None),
            **{f"layers.{2*i}.weight": (f"P_{i}", f"P_{i - 1}")
               for i in range(1, num_hidden_layers)},
            **{f"layers.{2*i}.bias": (f"P_{i}",)
               for i in range(num_hidden_layers)},
            f"layers.{2 * num_hidden_layers}.weight": (None, f"P_{num_hidden_layers - 1}"),
            f"layers.{2 * num_hidden_layers}.bias": (None,),
        })
    else:
        return permutation_spec_from_axes_to_perm({
            "layer0.weight": ("P_0", None),
            **{f"layer{i}.weight": (f"P_{i}", f"P_{i - 1}")
               for i in range(1, num_hidden_layers)},
            f"layer{num_hidden_layers}.weight": (None, f"P_{num_hidden_layers - 1}"),
        })


class NeuralAlignDiff:
    def perm_coord_descent(self, net0, net1, epochs=1, device=None):
        with torch.no_grad():
            weights0 = [
                net0.layers[layer_i].weight.clone().cpu() for i, layer_i in enumerate(self.layer_indices)
            ]
            weights1 = [
                net1.layers[layer_i].weight.clone().cpu() for i, layer_i in enumerate(self.layer_indices)
            ]

            biases0 = [
                net0.layers[layer_i].bias.clone().cpu() for i, layer_i in enumerate(self.layer_indices)
            ]
            biases1 = [
                net1.layers[layer_i].bias.clone().cpu() for i, layer_i in enumerate(self.layer_indices)
            ]

            perm_mats = [None for _ in range(len(weights0))]

            for i in range(len(weights0)):
                perm_mats[i] = torch.eye(
                    weights0[i].shape[0]
                ).cpu()

            new_perm_mats = copy.deepcopy(perm_mats)

            print(len(new_perm_mats))
            for iteration in range(2):
                rperm = torch.Tensor([4, 1, 3, 2, 0]).long()
                progress = False

                for i in rperm:
                    obj = torch.zeros(
                        (
                            weights0[i].shape[0],
                            weights0[i].shape[0]
                        )
                    )

                    if i > 0:
                        obj += weights0[i] @ new_perm_mats[i - 1] @ weights1[i].T

                        # print(torch.sum(weights0[i] @ new_perm_mats[i - 1] @ weights1[i].T))

                    if i == 0:
                        obj += weights0[i] @  weights1[i].T

                        # print(torch.sum(weights0[i] @ weights1[i].T))

                    obj += torch.outer(biases0[i], biases1[i])
                    # print(torch.sum(torch.outer(biases0[i], biases1[i])))

                    if i < len(weights0) - 1:
                        obj += weights0[i + 1].T @ new_perm_mats[i + 1] @ weights1[i + 1]

                        # print(torch.sum(weights0[i + 1].T @ new_perm_mats[i + 1] @ weights1[i + 1]))

                    # print("................")


                    ri, ci = scipy.optimize.linear_sum_assignment(obj.detach().numpy(), maximize=True)
                    assert (torch.tensor(ri) == torch.arange(len(ri))).all()
                    oldL = torch.einsum('ij,ij->i', obj, torch.eye(weights0[i].shape[0])[self.permmat_to_perm(new_perm_mats[i]).long(), :]).sum()
                    newL = torch.einsum('ij,ij->i', obj, torch.eye(weights0[i].shape[0])[ci, :]).sum()
                    print(f"{iteration}/{i}: {newL - oldL}")
                    progress = progress or newL > oldL + 1e-12

                    new_perm_mats[i] = copy.deepcopy(self.perm_to_permmat(ci))

                if not progress:
                    print("NO", iteration)
                    break

            return [self.permmat_to_perm(new_perm_mats[i].long()) for i in range(len(new_perm_mats))]

    def __init__(self, model_cls, match_method, loader0, loader1, loaderc) -> None:
        self.permutations   = list()
        self.statistics     = list()
        self.layer_indices  = list()
        self.perms_calc     = False
        self.stats_calc     = False
        self.layers_indexed = False

        self.loader0         = loader0
        self.loader1         = loader1
        self.loaderc         = loaderc
        self.model_cls       = model_cls
        self.match_method = match_method

        self.dbg0 = None

    def run_act_similarity(self, net0, net1, epochs=1, loader=None, device=None):
        n = epochs * len(loader)
        F = None

        with torch.no_grad():
            net0.eval()
            net1.eval()

            for _ in range(epochs):
                for i, (images, _) in enumerate(tqdm(loader)):
                    img_t = images.float().to(device)
                    out0 = net0(img_t)
                    out0 = out0.reshape(out0.shape[0], out0.shape[1], -1).permute(0, 2, 1)
                    out0 = out0.reshape(-1, out0.shape[2]).float()

                    out1 = net1(img_t)
                    out1 = out1.reshape(out1.shape[0], out1.shape[1], -1).permute(0, 2, 1)
                    out1 = out1.reshape(-1, out1.shape[2]).float()

                    prod = out0.T @ out1

                    if i == 0:
                        F = torch.zeros(prod.shape).to(device)

                    F += prod

        return F

    def run_corr_matrix(self, net0, net1, epochs=1, loader=None, device=None):
        n = epochs * len(loader)
        mean0 = mean1 = std0 = std1 = outer = None
        with torch.no_grad():
            net0.eval()
            net1.eval()

            for _ in range(epochs):
                for i, (images, _) in enumerate(tqdm(loader)):
                    img_t = images.float().to(device)
                    out0 = net0(img_t)
                    out0 = out0.reshape(out0.shape[0], out0.shape[1], -1).permute(0, 2, 1)
                    out0 = out0.reshape(-1, out0.shape[2]).float()

                    out1 = net1(img_t)
                    out1 = out1.reshape(out1.shape[0], out1.shape[1], -1).permute(0, 2, 1)
                    out1 = out1.reshape(-1, out1.shape[2]).float()

                    mean0_b = out0.mean(dim=0)
                    mean1_b = out1.mean(dim=0)
                    std0_b = out0.std(dim=0)
                    std1_b = out1.std(dim=0)
                    outer_b = (out0.T @ out1) / out0.shape[0]

                    if i == 0:
                        mean0 = torch.zeros_like(mean0_b)
                        mean1 = torch.zeros_like(mean1_b)
                        std0 = torch.zeros_like(std0_b)
                        std1 = torch.zeros_like(std1_b)
                        outer = torch.zeros_like(outer_b)

                    mean0 += mean0_b / n
                    mean1 += mean1_b / n
                    std0 += std0_b / n
                    std1 += std1_b / n
                    outer += outer_b / n

        cov = outer - torch.outer(mean0, mean1)
        corr = cov / (torch.outer(std0, std1) + 1e-4)

        return corr

    def solve_lap(self, corr_mtx):
        corr_mtx_a = corr_mtx.cpu().numpy()
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(corr_mtx_a, maximize=True)

        assert (row_ind == np.arange(len(corr_mtx_a))).all()

        perm_map = torch.tensor(col_ind).long()

        return perm_map

    def perm_to_permmat(self, permutation):
        perm_mat = torch.zeros((len(permutation), len(permutation)))
        perm_mat[torch.arange(len(permutation)), permutation] = 1

        return perm_mat

    def permmat_to_perm(self, permmat):
        perm = torch.Tensor(permmat.shape[0])
        perm = torch.where(permmat == 1)[1]

        return perm

    def get_layer_perm(self, net0, net1, epochs=1, loader=None, device=None):
        corr_mtx = self.run_corr_matrix(
            net0,
            net1,
            epochs=1,
            loader=loader,
            device=device
        )

        return self.solve_lap(corr_mtx), corr_mtx

    def index_layers(self, model):
        if self.layers_indexed is True:
            return

        for idx, m in enumerate(model.layers):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                self.layer_indices.append(idx)

        print(self.layer_indices)
        self.layers_indexed = True

    def align_networks_smart(self, model0, model1, loader=None, device=None):
        cl0 = copy.deepcopy(model0.to("cpu")).to(device)
        cl1 = copy.deepcopy(model1.to("cpu")).to(device)

        self.index_layers(model0)

        if self.perms_calc is False:
            # self.permutations = self.perm_coord_descent(cl0, cl1, epochs=400, device=device)
            # self.perms_calc = True

            dct0 = copy.deepcopy(cl0.cpu().state_dict())
            dct1 = copy.deepcopy(cl1.cpu().state_dict())

            torch.manual_seed(0)
            iterations = 100
            global_perms = [
                torch.randperm(len(self.layer_indices) - 1) for _ in range(iterations)
            ]

            # ps = mlp_permutation_spec(5, True)
            # self.permutations, _ = weight_matching_ref(ps, dct0, dct1, max_iter=iterations, debug_perms=global_perms, legacy=False)
            # for perm in self.permutations:
            #     print(perm[:11])
            # # print("....................")
            # self.permutations = WeightMatching(epochs=iterations, debug=True, debug_perms=global_perms,)(self.layer_indices, cl0, cl1)
            # for perm in self.permutations:
            #     print(perm[:11])
            # self.permutations =  self.permutations + [self.permmat_to_perm(torch.eye(128))]


            # ps = mlp_permutation_spec(5, True)
            # cl0 = cl0.to("mps")
            # cl1 = cl1.to("mps")
            # self.permutations = wm_learning(cl0, cl1, self.loaderc, ps, dbg_perm=global_perms)
            # self.permutations =  self.permutations + [self.permmat_to_perm(torch.eye(128))]

            cl0 = copy.deepcopy(model0.to("cpu")).to(device)
            cl1 = copy.deepcopy(model1.to("cpu")).to(device)
            ste_matchin = SteMatching(
                torch.nn.CrossEntropyLoss(),
                self.loaderc,
                0.1,
                epochs=10,
                device="mps",
                wm_kwargs={
                    "epochs": 100,
                    "debug": False,
                    "debug_perms": global_perms
                }
            )
            self.permutations = ste_matchin(self.layer_indices, cl0, cl1)
            self.permutations =  self.permutations + [self.permmat_to_perm(torch.eye(128))]


        model0 = model0.to("mps")
        model1 = model1.to("mps")

        # self.permutations = list()
        # if self.perms_calc is False:
        #     for i, layer_idx in enumerate(self.layer_indices):
        #         if i == len(self.layer_indices) - 1:
        #             self.permutations.append(self.permmat_to_perm(torch.eye(128)))
        #             break

        #         perm_map_, corr_mtx = self.get_layer_perm(
        #             model0.subnet(cl0, layer_i=layer_idx).to("mps"),
        #             model1.subnet(cl1, layer_i=layer_idx).to("mps"),
        #             epochs=1,
        #             loader=loader,
        #             device=device
        #         )
        #         self.permutations.append(perm_map_)

        # for perm in self.permutations:
        #     print(perm[:11])

        print("DONE")
        last_perm_map = None
        for i, layer_idx in enumerate(self.layer_indices):
            perm_map = self.permutations[i]
            weight   = model1.layers[layer_idx].weight
            bias     = model1.layers[layer_idx].bias

            model1.layers[layer_idx].weight.data = weight[perm_map].clone()
            model1.layers[layer_idx].bias.data   = bias[perm_map].clone()

            weight = model1.layers[layer_idx].weight
            
            if i > 0:
                model1.layers[layer_idx].weight.data = weight[:, last_perm_map].clone()

            last_perm_map = perm_map

        # last_perm_map = self.permutations[-1]
        # weight = model1.classifier.weight
        # model1.classifier.weight.data = weight[:, last_perm_map].clone()

        # weight = model1.layers[self.layer_indices[-1]].weight
        # model1.layers[self.layer_indices[-1]].weight.data = weight[:, last_perm_map].clone()

        self.perms_calc = True

        return model0, model1

    def wrap_layers_smart(self, model, rescale):
        wrapped_model = model

        for i, layer_idx in enumerate(self.layer_indices):
            layer = model.layers[layer_idx]

            wrapper = LayerWrapper
            if isinstance(layer, nn.Conv2d):
                wrapper = LayerWrapper2D

            wrapped_model.layers[layer_idx] = wrapper(layer, rescale=rescale)

        return wrapped_model


    def mix_weights(self, model, model0, model1, alpha):
        sd0 = model0.state_dict()
        sd1 = model1.state_dict()
        sd_alpha = {k: (1 - alpha) * sd0[k] + alpha * sd1[k]
                    for k in sd0.keys()}

        model.load_state_dict(sd_alpha)


    def fuse_networks(self, model_args, model0, model1, alpha, loader=None, device=None, new_stats=True, permute = True):
        modela = self.model_cls(**model_args).to(device)

        if permute is True:
            model0, model1 = self.align_networks_smart(
                model0,
                model1,
                loader=self.loaderc,
                device=device
            )
            self.dbg0 = model0

        self.mix_weights(modela, model0, model1, alpha)

        if new_stats is False:
            return modela

        print("HERE")
        return self.REPAIR_smart(alpha, model0, model1, modela, loader=loader, device=device)

    def REPAIR_smart(self, alpha, model0, model1, modela, loader=None, device=None):
        model0_tracked = self.wrap_layers_smart(model0, rescale=False).to(device)
        model1_tracked = self.wrap_layers_smart(model1, rescale=False).to(device)
        modela_tracked = self.wrap_layers_smart(modela, rescale=False).to(device)

        if self.stats_calc is False:
            model0_tracked.train()
            model1_tracked.train()
            for m in model0_tracked.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.momentum = None
                    m.reset_running_stats()

            for m in model1_tracked.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.momentum = None
                    m.reset_running_stats()

            with torch.no_grad():
                for inputs, labels in self.loader0:
                    _ = model0_tracked(inputs.to(device))

                for inputs, labels in self.loader1:
                    _ = model1_tracked(inputs.to(device))

            model0_tracked.eval()
            model1_tracked.eval()

            for i, layer_idx in enumerate(self.layer_indices):
                layer0 = model0_tracked.layers[layer_idx]
                layer1 = model1_tracked.layers[layer_idx]

                stats0_ = layer0.get_stats()
                stats1_ = layer1.get_stats()

                self.statistics.append((stats0_, stats1_))

        for i, layer_idx in enumerate(self.layer_indices):
            stats0 = self.statistics[i][0]
            stats1 = self.statistics[i][1]

            mean = (1.0 - alpha) * stats0[0] + alpha * stats1[0]
            std = ((1.0 - alpha) * stats0[1].sqrt() + alpha * stats1[1].sqrt()).square()

            modela_tracked.layers[layer_idx].set_stats(mean, std)
            modela_tracked.layers[layer_idx].rescale = True

        modela_tracked.train()
        for m in modela_tracked.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.momentum = None
                m.reset_running_stats()

        with torch.no_grad():
            for _ in range(3):
                for inputs, labels in self.loaderc:
                    _ = modela_tracked(inputs.to(device))

        modela_tracked.eval()

        self.stats_calc = True

        return modela_tracked
