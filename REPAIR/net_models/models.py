import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, channels=128, layers=3, classes=10, bnorm=False):
        super().__init__()
        
        self.bnorm      = bnorm
        self.classes    = classes
        self.channels   = channels
        self.num_layers = layers
        self.subnet     = Subnet
        self.perm_spec  = MLPSpec(layers, bnorm=bnorm)
        
        mid_layers = [
            nn.Linear(28 * 28, channels, bias=True),
            nn.ReLU()
        ]
        for i in range(layers):
            lst  = [
                nn.Linear(channels, channels, bias=True),
                nn.ReLU(),
            ]
            if i == self.num_layers - 1:
                lst = [
                    nn.Linear(channels, channels, bias=True),
                ]
            mid_layers.extend(lst)
            
        self.layers = nn.Sequential(*mid_layers)

    def forward(self, x):
        if x.size(1) == 3:
            x = x.mean(1, keepdim=True)

        x = x.reshape(x.size(0), -1)
        x = self.layers(x)
 
        return x
    

class VGG(nn.Module):
    def __init__(self, cfg, w=1, classes=10, in_channels=3, bnorm=False):
        super().__init__()

        self.in_channels = in_channels
        self.w           = w
        self.bnorm       = bnorm
        self.classes     = classes
        self.subnet      = VGGSubnet
        self.perm_spec   = VGGSpec(cfg, bnorm=bnorm)
        self.layers      = self._make_layers(cfg)

    def forward(self, x):
        out = self.layers[:-1](x)
        out = out.view(out.size(0), -1)
        out = self.layers[-1](out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = self.in_channels

        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers.append(nn.Conv2d(in_channels if in_channels == 3 else self.w*in_channels,
                                     self.w*x, kernel_size=3, padding=1))
                
                if self.bnorm is True:
                    layers.append(nn.BatchNorm2d(self.w*x))

                layers.append(nn.ReLU(inplace=True))
                in_channels = x

        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        layers += [nn.Linear(self.w * cfg[-2], self.classes)]

        return nn.Sequential(*layers)


class Subnet(nn.Module):
    def __init__(self, model, layer_i):
        super().__init__()
        self.model = model
        self.layer_i = layer_i

    def forward(self, x):
        if x.size(1) == 3:
            x = x.mean(1, keepdim=True)

        x = x.reshape(x.size(0), -1)
        x = self.model.layers[:self.layer_i + 1](x)
        
        return x
    

class VGGSubnet(nn.Module):
    def __init__(self, model, layer_i):
        super().__init__()
        self.model = model
        self.layer_i = layer_i

    def forward(self, x):
        x = self.model.layers[:self.layer_i + 1](x)
        
        return x
    

class SigmaWrapper(torch.nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.hess  = None
        self.cnt   = None

    def get_stats(self):
        assert self.hess is not None
        
        return self.hess / self.cnt

    def forward(self, x):
        hess = torch.multiply(x.unsqueeze(-1), x.unsqueeze(1)).sum(0)
        hess += torch.eye(hess.shape[0]).to(x.device) * 0.0001

        if self.hess is None:
            self.cnt  = 0
            self.hess = torch.zeros_like(hess)

        self.hess += hess
        self.cnt += x.shape[0]

        x = self.layer(x)

        return x
    

class CovWrapper(torch.nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.cov   = None

    def get_stats(self):
        assert self.cov is not None
        
        return self.cov

    def forward(self, x):
        self.cov = torch.cov(x)

        x = self.layer(x)

        return x
    

class LayerWrapper(nn.Module):
    def __init__(self, layer, rescale=False, w=False):
        super().__init__()
        self.layer   = layer
        self.rescale = rescale

        if w is True:
            self.bn = nn.BatchNorm1d(len(layer.layer.weight))
            self.bn.to(self.layer.layer.weight.device)
        else:
            self.bn = nn.BatchNorm1d(len(layer.weight))
            self.bn.to(self.layer.weight.device)

    def get_stats(self):
        mean = self.bn.running_mean
        std  = self.bn.running_var

        return mean, std
    
    def set_stats(self, mean, var):
        self.bn.bias.data = mean
        self.bn.weight.data = (var + 1e-4).sqrt()

    def forward(self, x):
        x = self.layer(x)
        x_rescaled = self.bn(x)

        if self.rescale is True:
            return x_rescaled
        
        return x
    

class LayerWrapper2D(nn.Module):
    def __init__(self, layer, rescale=False, w=False):
        super().__init__()
        self.layer   = layer
        self.rescale = rescale

        if w is True:
            self.bn = nn.BatchNorm2d(self.layer.layer.weight.shape[0])
            self.bn.to(self.layer.layer.weight.device)
        else:
            self.bn = nn.BatchNorm2d(layer.weight.shape[0])
            self.bn.to(self.layer.weight.device)

    def get_stats(self):
        mean = self.bn.running_mean
        std  = self.bn.running_var

        return mean, std
    
    def set_stats(self, mean, var):
        self.bn.bias.data = mean
        self.bn.weight.data = (var + 1e-7).sqrt()

    def forward(self, x):
        x = self.layer(x)
        x_rescaled = self.bn(x)

        if self.rescale is True:
            return x_rescaled
        
        return x


class VGGSpec:
    def __init__(self, cfg, bnorm=False):
        self._cfg       = cfg
        self._layer_spec = list()
        self._perm_spec = list()

        offset = 0
        i      = 0
        for c in self._cfg:
            if c == "M":
                offset += 1
                continue

            modules = [
                f"layers.{offset}.weight",
                f"layers.{offset}.bias",
            ]
            perms = [
                (i, i - 1),
                (i, -1)
            ]

            if bnorm is True:
                modules.extend(
                    [
                        f"layers.{offset + 1}.weight",
                        f"layers.{offset + 1}.bias",
                        f"layers.{offset + 1}.running_mean",
                        f"layers.{offset + 1}.running_var",
                    ]
                )

                perms.extend(
                    [
                        (i, -1),
                        (i, -1),
                        (i, -1),
                        (i, -1),
                    ]   
                )

            self._layer_spec.append(modules)
            self._perm_spec.append(perms)
            offset += 3 - (not bnorm)

            i += 1
        
        self._layer_spec.append(
            [
                f"layers.{offset + 1}.weight",
                f"layers.{offset + 1}.bias",
            ]
        )
        self._perm_spec.append(
            [
                (-1, i - 1),
                (-1, -1)
            ]
        )

    @property
    def cfg(self):
        return self._cfg
    
    @property
    def layer_spec(self):
        return self._layer_spec
    
    @property
    def perm_spec(self):
        return self._perm_spec
    

class MLPSpec:
    def __init__(self, layers, bnorm=False):
        self._cfg        = layers
        self._layer_spec = list()
        self._perm_spec  = list()

        offset = 0
        for i in range(layers):
            modules = [
                f"layers.{offset}.weight",
                f"layers.{offset}.bias",
            ]

            perms = [
                (i, i - 1),
                (i, -1)
            ]

            if bnorm is True:
                modules.extend(
                    [
                        f"layers.{offset + 1}.weight",
                        f"layers.{offset + 1}.bias",
                        f"layers.{offset + 1}.running_mean",
                        f"layers.{offset + 1}.running_var",
                        f"layers.{offset + 1}.num_batches_tracked",
                    ]
                )

                perms.extend(
                    [
                        (i, -1),
                        (i, -1),
                        (i, -1),
                        (i, -1),
                        (i, -1)
                    ]   
                )

            self._layer_spec.append(modules)
            self._perm_spec.append(perms)

            offset += 3 - (not bnorm)

        self._layer_spec.append(
            [
                f"layers.{offset}.weight",
                f"layers.{offset}.bias",
            ]
        )
        self._perm_spec.append(
            [
                (-1, layers - 1),
                (-1, -1)
            ]
        )

        print(self._perm_spec)

    @property
    def cfg(self):
        return self._cfg
    
    @property
    def layer_spec(self):
        return self._layer_spec
    
    @property
    def perm_spec(self):
        return self._perm_spec
    

def index_layers(model):
    layer_indices = list()

    for idx, m in enumerate(model.layers):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, LayerWrapper) or isinstance(m, LayerWrapper2D):
            layer_indices.append(idx)

    return layer_indices


def test_vgg_specs():
    cfg = {
        'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    }
    
    for bnorm in [True, False]:
        for key in cfg.keys():
            spec_list = list()
            spec = VGGSpec(cfg[key], bnorm=bnorm)
            for el in spec.perm_spec:
                spec_list.append(int(el[0].split(".")[1]))

            net = VGG(spec.cfg, bnorm=bnorm)
            indices = index_layers(net)

            for i in range(len(indices)):
                if indices[i] != spec_list[i]:
                    return False
            
    return True


def test_mlp_specs():
    for l in range(3, 7):
        spec_list = list()
        spec = MLPSpec(l, bnorm=False)
        for el in spec.layer_spec:
            spec_list.append(int(el[0].split(".")[1]))

        net = MLP(layers=l, bnorm=False)
        indices = index_layers(net)

        for i in range(len(indices)):
            if indices[i] != spec_list[i]:
                return False
    
    spec = MLPSpec(5, bnorm=False)

    for el in spec.layer_spec:
        print(el)

    for el in spec.perm_spec:
        print(el)
    
    return True


if __name__ == "__main__":
    # res = test_vgg_specs()
    # print(res)

    # res = test_mlp_specs()
    # print(res)

    cfg = {
        'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    }

    spec_list = list()
    spec = VGGSpec(cfg["VGG11"], bnorm=True)
    for el in spec.perm_spec:
        print(el)
        # spec_list.append(int(el[0].split(".")[1]))
    
    print("..........")
    for el in spec._layer_spec:
        print(el)