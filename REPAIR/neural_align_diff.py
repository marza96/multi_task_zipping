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

        # torch.manual_seed(0)
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

        # ps = vgg11_permutation_spec_bnorm()
        # ps = mlp_permutation_spec(5, True)
        # params_a = cl0.state_dict()
        # params_b = cl1.state_dict()
        # perms, _ = weight_matching_ref(ps, params_a, params_b, max_iter=300, debug_perms=dbg_perms, init_perm=init_perm_leg, legacy=False)
        # for perm in perms:
        #     print(perm[:11])

        # print("   ")
        # print(".............")
        # print("   ")

        ### NOTE My Weight Matching Gen
        # from .matching.weight_matching_gen import WeightMatching
        # wm = WeightMatching(epochs=iters, debug_perms=dbg_perms, ret_perms=False)
        # cl0, cl1 = wm(self.layer_indices_ext, cl0, cl1, init_perm=init_perm)
        
        from .matching.weight_matching import WeightMatching
        wm = WeightMatching(epochs=iters, debug_perms=dbg_perms, ret_perms=False)
        cl0, cl1 = wm(self.layer_indices, cl0, cl1, init_perm=init_perm)

        return cl0, cl1
        # from .matching.matching_utils import permmat_to_perm, apply_permutation
        # from .matching.weight_matching_gen import WeightMatching

        # wm = WeightMatching(epochs=iters, debug_perms=dbg_perms, ret_perms=True)
        # from .matching.ste_weight_matching_gen import SteMatching
        # sm = SteMatching(torch.nn.functional.cross_entropy, self.loaderc, 0.25, wm, debug=dbg, epochs=20, device="mps")
        # sm(self.layer_indices_ext, cl0, cl1)
    

        # from .matching.legacy_weight_matching import vgg11_permutation_spec_bnorm, vgg11_permutation_spec, weight_matching_ref
        # from .matching.legacy_weight_matching import wm_learning, weight_matching_ref
        # from .matching.matching_utils import permmat_to_perm, apply_permutation

        # ps = vgg11_permutation_spec_bnorm()
        # perms = wm_learning(model0.to("mps"), model1.to("mps"), self.loaderc, ps, "mps", 0.25, dbg_perm=dbg_perms, debug=dbg, epochs=1)
        # perms =  perms + [permmat_to_perm(torch.eye(10))]
        # cl1 = apply_permutation(self.layer_indices, cl1.to("cpu"), perms)
        # return cl0, cl1
    

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

        return self.REPAIR_smart(alpha, self.aligned0, self.aligned1, modela, loader=loader, device=device)
    
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
            self.reset_bn_stats(model0_tracked, self.loader0)
            self.reset_bn_stats(model1_tracked, self.loader1)

            # for i, layer_idx in enumerate(self.layer_indices):
            #     layer = model0_tracked.layers[layer_idx]

            #     try:
            #         layer.layer.bn.eval()
            #     except:
            #         pass

            #     layer.bn.train()
            #     layer.bn.momentum = None
            #     layer.bn.reset_running_stats()

            # for i, layer_idx in enumerate(self.layer_indices):
            #     layer = model1_tracked.layers[layer_idx]

            #     try:
            #         layer.layer.bn.eval()
            #     except:
            #         pass

            #     layer.bn.train()
            #     layer.bn.momentum = None
            #     layer.bn.reset_running_stats()

            # with torch.no_grad():
            #     for inputs, labels in tqdm.tqdm(self.loader0):
            #         _ = model0_tracked(inputs.to(device))

            #     for inputs, labels in tqdm.tqdm(self.loader1):
            #         _ = model1_tracked(inputs.to(device))

            # model0_tracked.eval()
            # model1_tracked.eval()

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

        self.reset_bn_stats(modela_tracked, self.loaderc)
        
        # for i, layer_idx in enumerate(self.layer_indices):
        #     layer = modela_tracked.layers[layer_idx]
            
        #     try:
        #         layer.layer.bn.eval()
        #     except:
        #         pass

        #     layer.bn.train()
        #     layer.bn.momentum = None
        #     layer.bn.reset_running_stats()

        # with torch.no_grad():
        #     for _ in range(1):
        #         for inputs, labels in tqdm.tqdm(self.loaderc):
        #             _ = modela_tracked(inputs.to(device))

        modela_tracked.eval()

        self.stats_calc = True

        return modela_tracked
