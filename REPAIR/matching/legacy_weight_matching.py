import torch
import scipy
import copy
import tqdm

from typing import NamedTuple
from collections import defaultdict, OrderedDict


'''
    Legacy code From the autors of <<PAPER NAME>>
    Modified and used for research purposes
'''


class PermutationSpec(NamedTuple):
    perm_to_axes: dict
    axes_to_perm: dict


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
    

def get_permuted_param(ps, perm, k: str, params, except_axis=None):
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


def apply_permutation(ps, perm, params):
    """Apply a `perm` to `params`."""
    ret = {}
    for k in params.keys():
        if params[k].dim() != 0:  # avoid num_batches_tracked
            ret[k] = get_permuted_param(ps, perm, k, params)
        else:
            ret[k] = params[k]

    return ret


def weight_matching_ref(ps, params_a, params_b, max_iter=300, debug_perms=None, init_perm=None, print_flg=False, legacy=True):
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
        return perm, final_perm
    
    return final_perm, metrics


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

def wm_learning(model_a, model_b, train_loader, permutation_spec, device, dbg_perm=None):
    from torch.nn.utils.stateless import functional_call
    import torchopt

    best_perm = None
    best_perm_loss = 999
    perm = None

    train_state = model_a.state_dict()
    model_target = copy.deepcopy(model_a)

    for key in train_state:
        train_state[key] = train_state[key].float()

    for epoch in tqdm(range(0, 20)):
        correct = 0.
        loss_acum = 0.0
        total = 0

        for i, (x, t) in enumerate(tqdm(train_loader, leave=False)):
            x = x.to(device)
            t = t.to(device)

            # projection by weight matching
            perm, pdb = weight_matching_ref(permutation_spec,
                                        train_state, flatten_params(model_b),
                                        max_iter=100, debug_perms=dbg_perm, init_perm=perm)

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
            loss = torch.nn.functional.cross_entropy(output, t)
            loss.backward()

            for key in train_state.keys():
                new_param = train_state[key].detach() - 0.005 * train_state[key].grad.detach()
                train_state[key].data = new_param.detach()

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