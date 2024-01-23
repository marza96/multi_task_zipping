import torch
import copy
import tqdm
import torch.nn as nn

from .net_models.models import LayerWrapper, LayerWrapper2D, SigmaWrapper

torch.set_printoptions(precision=4, sci_mode=False)


class NeuralAlignDiff:
    def __init__(self, model_cls, match_method, loader0, loader1, loaderc) -> None:
        self.permutations      = list()
        self.statistics        = list()
        self.layer_indices     = list()
        self.layer_indices_ext = list()
        self.perms_calc        = False
        self.stats_calc        = False
        self.layers_indexed    = False

        self.loader0         = loader0
        self.loader1         = loader1
        self.loaderc         = loaderc
        self.model_cls       = model_cls
        self.match_method = match_method

        self.aligned0 = None
        self.aligned1 = None

    def index_layers(self, model):
        if self.layers_indexed is True:
            return

        for idx, m in enumerate(model.layers):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, LayerWrapper) or isinstance(m, LayerWrapper2D):
                self.layer_indices.append(idx)

        self.layers_indexed    = True
        self.layer_indices_ext = model.perm_spec

    def wrap_layers_sigma(self, model):
        wrapped_model = copy.deepcopy(model)

        for i, layer_idx in enumerate(self.layer_indices):
            layer = model.layers[layer_idx]

            wrapper = SigmaWrapper
            wrapped_model.layers[layer_idx] = wrapper(layer)

        return wrapped_model

    def align_networks_smart(self, model0, model1, device=None):
        cl0 = copy.deepcopy(model0.to("cpu")).to(device)
        cl1 = copy.deepcopy(model1.to("cpu")).to(device)
        
        self.index_layers(model0)

        dbg_perms = None
        dbg       = False
        iters     = 1000
        init_perm = None
        init_perm_leg = None

        torch.manual_seed(0)
        # dbg_perms = [
        #     torch.randperm(len(self.layer_indices) - 1) for _ in range(1000)
        # ]
        # dbg = True
        # iters = 1000

        # init_perm = [
        #     torch.randperm(128) for i in range(len(self.layer_indices) - 1)
        # ]
        # init_perm_leg = {("P_%d" % idx): init_perm[idx] for idx in range(len(init_perm))}

        # init_perm = None
        # init_perm_leg = None

        # from .matching.legacy_weight_matching import vgg11_permutation_spec_bnorm, vgg11_permutation_spec, weight_matching_ref
        # from .matching.legacy_weight_matching import wm_learning, weight_matching_ref
        # from .matching.matching_utils import permmat_to_perm, apply_permutation

        # ps = vgg11_permutation_spec_bnorm()
        # perms = wm_learning(model0.to("mps"), model1.to("mps"), self.loaderc, ps, "mps", 0.25, dbg_perm=dbg_perms, debug=dbg, epochs=1)
        # perms =  perms + [permmat_to_perm(torch.eye(10))]
        # cl1 = apply_permutation(self.layer_indices, cl1.to("cpu"), perms)
        # return cl0, cl1


        ### NOTE Legacy Weight Matching
        from .matching.legacy_weight_matching import mlp_permutation_spec, vgg11_permutation_spec, vgg11_permutation_spec_bnorm, weight_matching_ref
        from .matching.legacy_weight_matching import weight_matching_ref
        from .matching.matching_utils import permmat_to_perm, apply_permutation

        # ps = vgg11_permutation_spec_bnorm()
        # ps = mlp_permutation_spec(5, True)
        # params_a = cl0.state_dict()
        # params_b = cl1.state_dict()
        # perms, _ = weight_matching_ref(ps, params_a, params_b, max_iter=300, debug_perms=dbg_perms, init_perm=init_perm_leg, legacy=False)
        # perms =  perms + [permmat_to_perm(torch.eye(128))]
        # cl1 = apply_permutation(self.layer_indices, cl1.to("cpu"), perms)
        # return cl0.to(device), cl1.to(device)
        
        # # NOTE My Weight Matching Gen
        # from .matching.weight_matching_gen import WeightMatching
        # wm = WeightMatching(epochs=iters, debug_perms=dbg_perms, ret_perms=False)
        # cl0, cl1 = wm(self.layer_indices_ext, cl0, cl1, init_perm=init_perm)
        # return cl0.to(device), cl1.to(device)
        
        ## NOTE STE FROM ON NOW
        # from .matching.weight_matching_gen import WeightMatching
        # wm = WeightMatching(epochs=iters, debug_perms=dbg_perms, ret_perms=True)
        # from .matching.ste_weight_matching_gen import SteMatching
        # sm = SteMatching(torch.nn.functional.cross_entropy, self.loaderc, 0.25, wm, debug=dbg, epochs=30, device="mps")
        # cl0, cl1 = sm(self.layer_indices_ext, cl0, cl1)
        # print(cl1.layers[0].weight[:5, :5])
        # return cl0, cl1

        from .matching.legacy_weight_matching import vgg11_permutation_spec_bnorm, vgg11_permutation_spec, weight_matching_ref
        from .matching.legacy_weight_matching import wm_learning, weight_matching_ref
        from .matching.matching_utils import permmat_to_perm, apply_permutation
        ps = vgg11_permutation_spec_bnorm()
        ps = mlp_permutation_spec(5, True)
        perms = wm_learning(model0.to("mps"), model1.to("mps"), self.loaderc, ps, "mps", 0.25, dbg_perm=dbg_perms, debug=dbg, epochs=30)
        perms =  perms + [permmat_to_perm(torch.eye(128))]
        cl1 = apply_permutation(self.layer_indices, cl1.to("cpu"), perms)
        print(cl1.layers[0].weight[:5, :5])
        return cl0, cl1
    
        if self.perms_calc is True:
            return self.aligned0, self.aligned1

        if self.perms_calc is False:
            self.perms_calc = True

            cl0, cl1 = self.match_method(self.layer_indices, cl0, cl1)
            return cl0, cl1

    def wrap_layers_smart(self, model, rescale):
        wrapped_model = copy.deepcopy(model)

        for i, layer_idx in enumerate(self.layer_indices):
            layer = model.layers[layer_idx]

            w = False
            if isinstance(layer, LayerWrapper) or isinstance(layer, LayerWrapper2D):
                w = True

            wrapper = LayerWrapper
            if isinstance(layer, nn.Conv2d):
                wrapper = LayerWrapper2D

            wrapped_model.layers[layer_idx] = wrapper(layer, rescale=rescale, w=w)

        return wrapped_model

    def mix_weights(self, model, model0, model1, alpha):
        sd0 = model0.state_dict()
        sd1 = model1.state_dict()
        sd_alpha = {k: (1 - alpha) * sd0[k] + alpha * sd1[k]
                    for k in sd0.keys()}

        model.load_state_dict(sd_alpha)

    def fuse_networks(self, model_args, model0, model1, alpha, loader=None, device=None, new_stats=True, permute=True, model_mod=None):
        modela = self.model_cls(**model_args).to(device)

        if model_mod is not None:
            modela = model_mod(modela)

        if permute is True:
            model0, model1 = self.align_networks_smart(
                model0,
                model1,
                device=device
            )

            self.aligned0 = model0
            self.aligned1 = model1

        self.mix_weights(modela, self.aligned0, self.aligned1, alpha)

        if new_stats is False:
            return modela
        
        final = repair(self.aligned0, self.aligned1, modela, self.model_cls, self.layer_indices, self.loader0, self.loader1, self.loaderc, device=device)
        print("REF", final.layers[2].bn.weight[:5])

        # final = self.REPAIR_smart(alpha, self.aligned0, self.aligned1, modela, loader=loader, device=device)
        # print("MY", final.layers[2].bn.weight[:5])
        
        return final
    
    def reset_bn_stats(self, model, loader, device="cpu", epochs=1):
        for m in model.modules():
            if type(m) == nn.BatchNorm2d:
                m.momentum = None 
                m.reset_running_stats()

        model.train()

        for _ in range(epochs):
            with torch.no_grad():
                for images, _ in tqdm.tqdm(loader):
                    output = model(images.to(device))

    def REPAIR_smart(self, alpha, model0, model1, modela, loader=None, device=None, wm=False):
        model0_tracked = self.wrap_layers_smart(model0, rescale=False).to(device)
        model1_tracked = self.wrap_layers_smart(model1, rescale=False).to(device)
        modela_tracked = self.wrap_layers_smart(modela, rescale=False).to(device)
        
        if self.stats_calc is False:
            self.reset_bn_stats(model0_tracked, self.loader0, device)
            self.reset_bn_stats(model1_tracked, self.loader1, device)

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

        self.reset_bn_stats(modela_tracked, self.loaderc, device)
        
        modela_tracked.eval()

        self.stats_calc = True

        return modela_tracked
    

class ResetLinear(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.h = h = layer.out_features if hasattr(layer, 'out_features') else 768
        self.layer = layer
        self.bn = nn.BatchNorm1d(h)
        self.rescale = False
        
    def set_stats(self, goal_mean, goal_var):
        self.bn.bias.data = goal_mean
        goal_std = (goal_var + 1e-5).sqrt()
        self.bn.weight.data = goal_std
        
    def forward(self, *args, **kwargs):
        x = self.layer(*args, **kwargs)
        if self.rescale:
            x = self.bn(x)
        else:
            self.bn(x)
        return x
    

def reset_bn_stats(model, loader, epochs=1, device="cpu"):
    # resetting stats to baseline first as below is necessary for stability
    for m in model.modules():
        if type(m) == nn.BatchNorm2d:
            m.momentum = None # use simple average
            m.reset_running_stats()

    # run a single train epoch with augmentations to recalc stats
    model.train()
    for _ in range(epochs):
        with torch.no_grad():
            for images, _ in loader:
                output = model(images.to(device))
    

def make_tracked_net(net, model_cls, layer_indices, device="cpu"):
    net1 = model_cls(layers=len(layer_indices) - 1, classes=net.classes).to(device)
    net1.load_state_dict(net.state_dict())
    for i in range(len(layer_indices)):
        idx = layer_indices[i]
        if isinstance(net1.layers[idx], nn.Linear):
            net1.layers[idx] = ResetLinear(net1.layers[idx])
    return net1.to(device).eval()


def repair(model0, model1, model_a, model_cls, layer_indices, loader0, loader1, loaderc, device="cpu"):
    ## calculate the statistics of every hidden unit in the endpoint networks
    ## this is done practically using PyTorch BatchNorm2d layers.
    wrap0 = make_tracked_net(model0, model_cls, layer_indices, device=device)
    wrap1 = make_tracked_net(model1, model_cls, layer_indices, device=device)
    reset_bn_stats(wrap0, loader0)
    reset_bn_stats(wrap1, loader1)

    wrap_a = make_tracked_net(model_a, model_cls, layer_indices, device=device)
    ## set the goal mean/std in added bns of interpolated network, and turn batch renormalization on
    for m0, m_a, m1 in zip(wrap0.modules(), wrap_a.modules(), wrap1.modules()):
        if not isinstance(m0, ResetLinear):
            continue
        # get goal statistics -- interpolate the mean and std of parent networks
        mu0 = m0.bn.running_mean
        mu1 = m1.bn.running_mean
        goal_mean = (mu0 + mu1)/2
        var0 = m0.bn.running_var
        var1 = m1.bn.running_var
        goal_var = ((var0.sqrt() + var1.sqrt())/2).square()
        # set these in the interpolated bn controller
        m_a.set_stats(goal_mean, goal_var)
        # turn rescaling on
        m_a.rescale = True
        
    # reset the tracked mean/var and fuse rescalings back into conv layers 
    reset_bn_stats(wrap_a, loaderc)

    return wrap_a
