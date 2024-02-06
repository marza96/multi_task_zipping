from typing import Any
import torch
import copy
import tqdm
import time

from .matching_utils import perm_to_permmat, permmat_to_perm, solve_lap, apply_permutation


class WeightMatching():
    def __init__(self, debug=False, epochs=1, debug_perms=None, ret_perms=False, device="cpu"):
        self.debug       = debug
        self.epochs      = epochs
        self.debug_perms = debug_perms
        self.ret_perms   = ret_perms
        self.device      = device
        self.perms       = None
        self.iteration   = 0

    def auction_lap(self, X, eps, compute_score=False):
        """
            X: n-by-n matrix w/ integer entries
            eps: "bid size" -- smaller values means higher accuracy w/ longer runtime
        """
        # --
        # Init
        
        cost     = torch.zeros((1, X.shape[1])).to("cuda")
        curr_ass = (torch.zeros(X.shape[0]).long() - 1).to("cuda")
        bids     = torch.zeros(X.shape).to("cuda")
        
        # if X.is_cuda:
        #     cost, curr_ass, bids = cost.cuda(), curr_ass.cuda(), bids.cuda()
        
        
        counter = 0
        while (curr_ass == -1).any():
            counter += 1
            
            # --
            # Bidding
            
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
                
                # curr_ass[(curr_ass.view(-1, 1) == have_bidder.view(1, -1)).sum(dim=1)] = -1
                ind = (curr_ass.view(-1, 1) == have_bidder.view(1, -1)).sum(dim=1).nonzero()
                curr_ass[ind] = -1
                
                curr_ass[high_bidders] = have_bidder.squeeze()
            
        score = None
        if compute_score:
            score = int(X.gather(dim=1, index=curr_ass.view(-1, 1)).sum())
        
        return score, curr_ass, counter

    @staticmethod
    def get_permuted_param(param, perms, perm_axes, except_axis=None):
        w = param

        for ax_idx, p in enumerate(perm_axes):
            if ax_idx == except_axis:
                continue

            if p != -1:
                w = torch.index_select(w, ax_idx, perms[p].int())

        return w

    def objective_f(self, idx, perm_mats, state_dict0, state_dict1, spec, ste):
        mod_spec  = copy.deepcopy(spec.layer_spec[idx])
        perm_spec = copy.deepcopy(spec.perm_spec[idx])

        if idx < len(spec.layer_spec) - 1:
            mod_spec.extend(spec.layer_spec[idx + 1][:1])
            perm_spec.extend(spec.perm_spec[idx + 1][:1])

        cost = None
        size = len(self.perms[idx])

        for i in range(len(mod_spec)):
            axis   = perm_spec[i].index(int(idx))

            if ('running_mean' in mod_spec[i]) or ('running_var' in mod_spec[i]) or ('num_batches_tracked' in mod_spec[i]):
                continue
            
            w_0 = state_dict0[mod_spec[i]].clone().detach()
            w_1 = state_dict1[mod_spec[i]].clone().detach()

            w_1 = self.get_permuted_param(w_1, self.perms, perm_spec[i], except_axis=axis)

            w_0 = torch.moveaxis(w_0, axis, 0).reshape((size, -1))
            w_1 = torch.moveaxis(w_1, axis, 0).reshape((size, -1))

            c = w_0 @ w_1.T

            if cost is None:
                cost = torch.zeros((size, size)).to(self.device)

            cost += c

        return cost

    def objective(self, idx, perm_mats, state_dict0, state_dict1, net0, net1, spec, ste):
        l_type    = None

        layer_spec = spec.layer_spec
        layer_idx = int(layer_spec[idx][0].split(".")[1])

        if ste is True:
            l_type = net0.layers[layer_idx].layer_hat
        else:
            l_type = net0.layers[layer_idx]

        # assert l_type is not None

        return self.objective_f(idx, perm_mats, state_dict0, state_dict1, spec, ste)

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
                    net_state_dict[layer_key],
                    self.perms,
                    perm_axes,
                    except_axis=None
                )

                net_state_dict[layer_key] = perm_param

        net.load_state_dict(net_state_dict)

        return net

    def __call__(self, spec, net0, net1, ste=False, init_perm=None):
        with torch.no_grad():
            perm_specs  = spec.perm_spec
            layer_specs = spec.layer_spec
            w_shapes    = list()
            perm_mats   = [None for _ in range(len(perm_specs))]
            self.perms  = [None for _ in range(len(perm_specs))]

            for i in range(len(perm_specs)):
                layer_idx = int(layer_specs[i][0].split(".")[1])
                w_shape   = None

                if ste is True:
                    w_shape = net0.layers[layer_idx].layer_hat.weight.shape
                else:
                    w_shape = net0.layers[layer_idx].weight.shape

                assert w_shape is not None

                w_shapes.append(w_shape)

                perm_mats[i] = torch.eye(
                    w_shape[0]
                ).to(self.device)
                self.perms[i] = torch.arange(w_shape[0]).to(self.device)
            
            if init_perm is not None:
                for i in range(len(perm_specs) - 1):
                    perm_mats[i] = perm_to_permmat(init_perm[i])
                    self.perms[i] = init_perm[i]

            state_dict0 = net0.state_dict()
            state_dict1 = net1.state_dict()

            start_time = time.time()

            index_tens = list()
            
            for i in range(len(perm_specs)):
                index_tens.append(torch.eye(w_shapes[i][0]).to(self.device))

            for iteration in (range(self.epochs)):
                self.iteration = iteration
                progress       = False
                rperm          = torch.randperm(len(perm_specs) - 1)

                if self.debug_perms is not None:
                    rperm = self.debug_perms[iteration]

                for i in rperm:
                    obj  = self.objective(i, perm_mats, state_dict0, state_dict1, net0, net1, spec, ste)
               
                    _, perm, _ = self.auction_lap(obj, eps=5e-3)

                    oldL = torch.einsum(
                        'ij,ij->i',
                        obj,
                        index_tens[i][permmat_to_perm(perm_mats[i]).long(), :]
                    ).sum()
                    newL = torch.einsum(
                        'ij,ij->i',
                        obj,
                        index_tens[i][perm, :]
                    ).sum()

                    progress     = progress or newL > oldL + 1e-12
                    perm_mats[i] = perm_to_permmat(perm).clone()
                    self.perms[i] = perm.clone()

                    if self.debug is True:
                        print(f"{iteration}/{i}: {newL - oldL}")

                self.iteration += 1

                elapsed_time = time.time() - start_time
                iterations_per_second = (self.iteration + 1) / elapsed_time
                # print("it/s", iterations_per_second)
                
                if not progress:
                    break

            final_perms = [permmat_to_perm(perm_mats[i].long()) for i in range(len(perm_mats))]

            # print("MY PERMS")
            # for perm in final_perms:
            #     print(perm[:11])

            if self.ret_perms is True:
                return final_perms

            final_perms += [permmat_to_perm(torch.eye(w_shapes[-1][0]))]
            net1 = self.apply_permutation(spec, net1, final_perms)

            return net0.to(self.device), net1.to(self.device)
