import torch
import copy
import tqdm
import torch.nn as nn

from .net_models.mlp import LayerWrapper, LayerWrapper2D, SigmaWrapper
from .matching.weight_matching import WeightMatching
from .matching.weight_matching_sigma import WeightMatchingSigma
from .matching.activation_matching import ActivationMatching
from .matching.ste_weight_matching import SteMatching
from .matching.pgd_weight_matching import PGDMatching
from .matching.legacy_weight_matching import mlp_permutation_spec, wm_learning, weight_matching_ref

torch.set_printoptions(precision=4, sci_mode=False)


class NeuralAlignDiff:
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

        self.aligned0 = None
        self.aligned1 = None

    def index_layers(self, model):
        if self.layers_indexed is True:
            return

        for idx, m in enumerate(model.layers):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                self.layer_indices.append(idx)

        self.layers_indexed = True

    def wrap_layers_sigma(self, model):
        wrapped_model = copy.deepcopy(model)

        for i, layer_idx in enumerate(self.layer_indices):
            layer = model.layers[layer_idx]

            wrapper = SigmaWrapper
            wrapped_model.layers[layer_idx] = wrapper(layer)

        return wrapped_model

    def align_networks_smart(self, model0, model1, loader=None, device=None):
        cl0 = copy.deepcopy(model0.to("cpu")).to(device)
        cl1 = copy.deepcopy(model1.to("cpu")).to(device)

        self.index_layers(model0)

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

            self.aligned0 = model0
            self.aligned1 = model1

        self.mix_weights(modela, self.aligned0, self.aligned1, alpha)

        if new_stats is False:
            return modela

        return self.REPAIR_smart(alpha, self.aligned0, self.aligned1, modela, loader=loader, device=device)

    def REPAIR_smart(self, alpha, model0, model1, modela, loader=None, device=None, wm=False):
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
                for inputs, labels in tqdm.tqdm(self.loader0):
                    _ = model0_tracked(inputs.to(device))

                for inputs, labels in tqdm.tqdm(self.loader1):
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
            for _ in range(1):
                for inputs, labels in tqdm.tqdm(self.loaderc):
                    _ = modela_tracked(inputs.to(device))

        modela_tracked.eval()

        self.stats_calc = True

        return modela_tracked
