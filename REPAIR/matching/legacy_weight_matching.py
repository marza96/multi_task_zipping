import torch
import scipy
import copy
import tqdm

from typing import Any, NamedTuple
from collections import defaultdict, OrderedDict

from .matching_utils import apply_permutation, permmat_to_perm


'''
    Legacy code From the autors of
    <<Revisiting Permutation Symmetry for 
    Merging Models between Different Datasets>>
    Modified and used for research purposes
'''


class PermutationSpec(NamedTuple):
    perm_to_axes: dict
    axes_to_perm: dict


def permutation_spec_from_axes_to_perm(axes_to_perm: dict) -> PermutationSpec:
    perm_to_axes = defaultdict(list)
    # print(axes_to_perm)
    for wk, axis_perms in axes_to_perm.items():
        # print(axis_perms)
        for axis, perm in enumerate(axis_perms):
            if perm is not None:
                perm_to_axes[perm].append((wk, axis))
        
    return PermutationSpec(perm_to_axes=dict(perm_to_axes), axes_to_perm=axes_to_perm)


def mlp_permutation_spec(num_hidden_layers: int, bias_flg: bool) -> PermutationSpec:
    """We assume that one permutation cannot appear in two axes of the same weight array."""
    assert num_hidden_layers >= 1

    if bias_flg:
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
    

def vgg11_permutation_spec_bnorm() -> PermutationSpec:
    return permutation_spec_from_axes_to_perm({
                "layers.0.weight": ("P_0", None, None, None),
                "layers.0.bias": ("P_0", None),
                "layers.1.weight": ("P_0", None),
                "layers.1.bias": ("P_0", None),
                "layers.1.running_mean": ("P_0", None),
                "layers.1.running_var": ("P_0", None),
                "layers.1.num_batches_tracked": ("P_0", None),

                "layers.4.weight": ("P_1", "P_0", None, None),
                "layers.4.bias": ("P_1", None),
                "layers.5.weight": ("P_1", None),
                "layers.5.bias": ("P_1", None),
                "layers.5.running_mean": ("P_1", None),
                "layers.5.running_var": ("P_1", None),
                "layers.5.num_batches_tracked": ("P_1", None),

                "layers.8.weight": ("P_2", "P_1", None, None),
                "layers.8.bias": ("P_2", None),
                "layers.9.weight": ("P_2", None),
                "layers.9.bias": ("P_2", None),
                "layers.9.running_mean": ("P_2", None),
                "layers.9.running_var": ("P_2", None),
                "layers.9.num_batches_tracked": ("P_2", None),

                "layers.11.weight": ("P_3", "P_2", None, None),
                "layers.11.bias": ("P_3", None),
                "layers.12.weight": ("P_3", None),
                "layers.12.bias": ("P_3", None),
                "layers.12.running_mean": ("P_3", None),
                "layers.12.running_var": ("P_3", None),
                "layers.12.num_batches_tracked": ("P_3", None),

                "layers.15.weight": ("P_4", "P_3", None, None),
                "layers.15.bias": ("P_4", None),
                "layers.16.weight": ("P_4", None),
                "layers.16.bias": ("P_4", None),
                "layers.16.running_mean": ("P_4", None),
                "layers.16.running_var": ("P_4", None),
                "layers.16.num_batches_tracked": ("P_4", None),

                "layers.18.weight": ("P_5", "P_4", None, None),
                "layers.18.bias": ("P_5", None),
                "layers.19.weight": ("P_5", None),
                "layers.19.bias": ("P_5", None),
                "layers.19.running_mean": ("P_5", None),
                "layers.19.running_var": ("P_5", None),
                "layers.19.num_batches_tracked": ("P_5", None),

                "layers.22.weight": ("P_6", "P_5", None, None),
                "layers.22.bias": ("P_6", None),
                "layers.23.weight": ("P_6", None),
                "layers.23.bias": ("P_6", None),
                "layers.23.running_mean": ("P_6", None),
                "layers.23.running_var": ("P_6", None),
                "layers.23.num_batches_tracked": ("P_6", None),

                "layers.25.weight": ("P_7", "P_6", None, None),
                "layers.25.bias": ("P_7", None),
                "layers.26.weight": ("P_7", None),
                "layers.26.bias": ("P_7", None),
                "layers.26.running_mean": ("P_7", None),
                "layers.26.running_var": ("P_7", None),
                "layers.26.num_batches_tracked": ("P_7", None),

                "layers.30.weight": (None, "P_7"),
                "layers.30.bias": (None, None),
    })
    

def vgg11_permutation_spec() -> PermutationSpec:
    return permutation_spec_from_axes_to_perm({
        "layers.0.weight": ("P_0", None, None, None),
        "layers.0.bias": ("P_0", None),

        "layers.3.weight": ("P_1", "P_0", None, None),
        "layers.3.bias": ("P_1", None),

        "layers.6.weight": ("P_2", "P_1", None, None),
        "layers.6.bias": ("P_2", None),

        "layers.8.weight": ("P_3", "P_2", None, None),
        "layers.8.bias": ("P_3", None),

        "layers.11.weight": ("P_4", "P_3", None, None),
        "layers.11.bias": ("P_4", None),

        "layers.13.weight": ("P_5", "P_4", None, None),
        "layers.13.bias": ("P_5", None),

        "layers.16.weight": ("P_6", "P_5", None, None),
        "layers.16.bias": ("P_6", None),

        "layers.18.weight": ("P_7", "P_6", None, None),
        "layers.18.bias": ("P_7", None),

        "layers.22.weight": (None, "P_7"),
        "layers.22.bias": (None, None),
    })
    

def get_permuted_param(ps, perm, k: str, params, except_axis=None, dbg=False):
    """Get parameter `k` from `params`, with the permutations applied."""
    w = params[k]

    # if dbg is True:
    #     print("k", k)

    # if k == 'classifier.weight':  # to reshape because of input shape is 3x 96 x 96
    #     w = w.reshape(126, 512 * 4, 3, 3)
    try:
        # print("PERM", ps.axes_to_perm[k], except_axis)
        for axis, p in enumerate(ps.axes_to_perm[k]):
            if dbg is True:
                print("IN", p)

            # Skip the axis we're trying to permute.
            if axis == except_axis:
                continue

            # None indicates that there is no permutation relevant to that axis.
            if p is not None:
                # print("AX IDX", axis)
                w = torch.index_select(w, axis, perm[p].int())
        # if k == 'classifier.weight':
        #     w = w.reshape(126, -1)
    except KeyError:
        pass
    
    return w


def apply_permutation_legacy(ps, perm, params):
    """Apply a `perm` to `params`."""
    ret = {}
    for k in params.keys():
        if params[k].dim() != 0:  # avoid num_batches_tracked
            ret[k] = get_permuted_param(ps, perm, k, params)
        else:
            ret[k] = params[k]

    return ret


def weight_matching_ref(ps, params_a, params_b, max_iter=300, debug_perms=None, init_perm=None, print_flg=False, legacy=True, ret_perms=True, model_cls=None):
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
            # print("-------")
            # print("OUT", p)
            for wk, axis in ps.perm_to_axes[p]:  # layer loop
                if ('running_mean' not in wk) and ('running_var' not in wk) and ('num_batches_tracked' not in wk):
                    w_a = params_a[wk]  # target
                    w_b = get_permuted_param(ps, perm, wk, params_b, except_axis=axis, dbg=False)
                    
                    # if iteration == 1:
                    #     if "bias" not in wk:
                    #         print("K", w_b[:5, :5])
                    #     else:
                    #         print("K", w_b[:5])
                    #     print(wk, axis)
                    #     print("......")
                    
                    
                    w_a = torch.moveaxis(w_a, axis, 0).reshape((n, -1))
                    w_b = torch.moveaxis(w_b, axis, 0).reshape((n, -1))

                    A += w_a @ w_b.T  # A is cost matrix to assignment,
            
            #         if iteration == 0 and p == "P_2":
            #             print(wk, w_b.shape, axis)
            #             print("w", (w_a)[:5, :5])

            # if iteration == 1:
            #     print("COST", A[:5, :5])
            #     return            
            
            
            # print("------")
            ri, ci = scipy.optimize.linear_sum_assignment(A.detach().numpy(), maximize=True)
            assert (torch.tensor(ri) == torch.arange(len(ri))).all()
            oldL = torch.einsum('ij,ij->i', A, torch.eye(n)[perm[p].long()]).sum()
            newL = torch.einsum('ij,ij->i', A, torch.eye(n)[ci, :]).sum()
            if print_flg:
                print(f"{iteration}/{p}: {newL - oldL}")
            progress = progress or newL > oldL + 1e-12

            perm[p] = torch.Tensor(ci)

            # if iteration == 1:
            #     print("PERM", perm[p][:11])
            #     print("P", p)

            p_params_b = apply_permutation_legacy(ps, perm, params_b)
            # l2_dist = get_l2(params_a, p_params_b)
            # metrics['step'].append(step)
            # metrics['l2_dist'].append(l2_dist)
            # if iteration == 1:
            #     return
            
            step += 1
        if not progress:
            break

    if ret_perms is False:
        best_params = apply_permutation(ps, perm, params_b)
        b_mod = model_cls()
        b_mod.load_state_dict(best_params).to(device)

        return b_mod

    perm = {key: perm[key].to(device) for key in perm}  # to device
    final_perm = [None for _ in range(len(perm.keys()))]
    for key in perm.keys():
        idx = key.split("_")[-1]
        final_perm[int(idx)] = perm[key].long()

    params_a = {key: params_a[key].to(device) for key in params_a}  # to device
    params_b = {key: params_b[key].to(device) for key in params_b}  # to device

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

def wm_learning(model_a, model_b, train_loader, permutation_spec, device, lr, epochs=10, dbg_perm=None, debug=False):
    from torch.nn.utils.stateless import functional_call
    import torchopt

    best_perm = None
    best_perm_loss = 999
    perm = None

    train_state = model_a.state_dict()
    model_target = copy.deepcopy(model_a)

    for key in train_state:
        train_state[key] = train_state[key].float()

    cnt = 0
    for epoch in tqdm.tqdm(range(0, epochs)):
        correct = 0.
        loss_acum = 0.0
        total = 0

        for i, (x, t) in enumerate(tqdm.tqdm(train_loader, leave=False)):
            x = x.to(device)
            t = t.to(device)
            cnt += 1

            # projection by weight matching
            perm, pdb = weight_matching_ref(permutation_spec,
                                        train_state, flatten_params(model_b),
                                        max_iter=100, debug_perms=dbg_perm, init_perm=perm)

            projected_params = apply_permutation_legacy(permutation_spec, perm, flatten_params(model_b))
            
            # ste
            for key in train_state:
                if ('running_mean' not in key) and ('running_var' not in key) and ('num_batches_tracked' not in key):
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

            if debug is True:
                if cnt == 200:
                    print("iter %d ....................." % i)
                    for key in train_state:
                        if not "layers.4" in key:
                            continue

                        try:
                            print("LEG G", key, train_state[key].grad[:5, :5])
                        except:
                            print("LEG G", key, train_state[key].grad[:5])

                    print("......................................")

            for key in train_state.keys():
                if ('running_mean' not in key) and ('running_var' not in key) and ('num_batches_tracked' not in key):
                    new_param = train_state[key].detach() -  lr * train_state[key].grad.detach()
                    train_state[key].data = new_param.detach()

            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(t.view_as(pred)).sum().item()

            if loss < best_perm_loss:
                best_perm_loss = loss.item()
                best_perm = perm
            
            loss_acum += loss.mean()
            total += 1

            if debug is True:
                if cnt == 200:
                    for p in pdb:
                        print(p[:11])

                    return

        print("LOSS: %d" % i, loss_acum / total)
    
    print("LEG BEST PERM")
    for key in best_perm.keys():
        print(best_perm[key][:11])

    final_perm = [None for _ in range(len(best_perm.keys()))]
    for key in best_perm.keys():
        idx = key.split("_")[-1]
        final_perm[int(idx)] = best_perm[key].long()

    return final_perm


class LegacyWeightMatching:
    def __init__(self, num_layers, max_iter=1000, net_type="mlp", device="cpu"):
        self.ps       = None
        self.device   = device
        self.max_iter = max_iter
        if net_type == "mlp":
            self.ps = mlp_permutation_spec(num_layers, True)

    def __call__(self, cl0, cl1) -> Any:
        dct0 = copy.deepcopy(cl0.cpu().state_dict())
        dct1 = copy.deepcopy(cl1.cpu().state_dict())

        perms, _ = weight_matching_ref(
            self.ps, 
            dct0, 
            dct1, 
            max_iter=self.max_iter, 
            legacy=False
        )

        self.permutations =  self.permutations + [permmat_to_perm(torch.eye(128))]
        cl1 = apply_permutation(self.layer_indices, cl1, self.permutations, device)
        
        return cl0.to(self.device), cl1.to(self.device)

       
        