import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, channels=128, layers=3, classes=10):
        super().__init__()

        self.classes    = classes
        self.channels   = channels
        self.num_layers = layers
        self.subnet     = Subnet
        
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
    def __init__(self, cfg, w=1, classes=10, in_channels=3):
        super().__init__()

        self.in_channels = in_channels
        self.w           = w
        self.classes     = classes
        self.subnet      = VGGSubnet
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
    