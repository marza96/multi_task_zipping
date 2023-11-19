import torch
import torch.nn as nn
import torch.nn.functional as F


# class MLP(nn.Module):
#     def __init__(self, h=128, layers=3):
#         super().__init__()

#         self.h          = h
#         self.num_layers = layers
#         self.subnet     = Subnet
#         self.fc1        = nn.Linear(28*28, h, bias=True)
        
#         mid_layers = []
#         for _ in range(layers):
#             mid_layers.extend([
#                 nn.Linear(h, h, bias=True),
#                 nn.ReLU(),
#             ])
            
#         self.layers = nn.Sequential(*mid_layers)
#         self.fc2 = nn.Linear(h, 10)

#     def forward(self, x):
#         if x.size(1) == 3:
#             x = x.mean(1, keepdim=True)

#         x = x.reshape(x.size(0), -1)
#         x = F.relu(self.fc1(x))
#         x = self.layers(x)
#         x = self.fc2(x)

#         return x


class MLP(nn.Module):
    def __init__(self, channels=128, layers=3, classes=10):
        super().__init__()

        self.classes    = classes
        self.channels   = channels
        self.num_layers = layers
        self.subnet     = Subnet
        self.fc1        = nn.Linear(28*28, channels, bias=True)
        
        mid_layers = []
        for _ in range(layers):
            mid_layers.extend([
                nn.Linear(channels, channels, bias=True),
                nn.ReLU(),
            ])
            
        self.layers = nn.Sequential(*mid_layers)
        self.fc2 = nn.Linear(channels, classes)

    def forward(self, x):
        if x.size(1) == 3:
            x = x.mean(1, keepdim=True)

        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.layers(x)
        x = self.fc2(x)

        return x
    

class CNN(nn.Module):
    def __init__(self, h=128, layers=5):
        super().__init__()
        
        self.h          = h
        self.num_layers = layers
        self.subnet     = CNNSubnet
        self.fc1        = nn.Conv2d(1, 64, 3, 1)

        mid_layers = []
        for _ in range(layers):
            mid_layers.extend([
                nn.Conv2d(64, 64, 3, 1),
                nn.ReLU(),
            ])

        self.layers = nn.Sequential(*mid_layers)
        self.fc2 = nn.Linear(64 * (28 - 2 * (layers + 1)) ** 2, 10)

    def forward(self, x):
        if x.size(1) == 3:
            x = x.mean(1, keepdim=True)

        x = F.relu(self.fc1(x))
        x = self.layers(x)
        x = torch.flatten(x, 1)
        x = self.fc2(x)

        return x


class CNNSubnet(nn.Module):
    def __init__(self, model, layer_i):
        super().__init__()
        self.model = model
        self.layer_i = layer_i

    def forward(self, x):
        if x.size(1) == 3:
            x = x.mean(1, keepdim=True)

        x = F.relu(self.model.fc1(x))
        x = self.model.layers[:2 * self.layer_i](x)
        
        return x


class Subnet(nn.Module):
    def __init__(self, model, layer_i):
        super().__init__()
        self.model = model
        self.layer_i = layer_i

    def forward(self, x):
        if x.size(1) == 3:
            x = x.mean(1, keepdim=True)

        x = x.reshape(x.size(0), -1)
        x = F.relu(self.model.fc1(x))
        x = self.model.layers[:2 * self.layer_i](x)
        
        return x
    

class LayerWrapper(nn.Module):
    def __init__(self, layer, rescale=False):
        super().__init__()
        self.layer   = layer
        self.rescale = rescale
        self.bn      = nn.BatchNorm1d(len(layer.weight))

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
    

class LayerWrapper2D(nn.Module):
    def __init__(self, layer, rescale=False):
        super().__init__()
        self.layer   = layer
        self.rescale = rescale
        self.bn      = nn.BatchNorm2d(self.layer.weight.shape[0])

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
    
def main():
    cnn = CNN()
    tens = torch.zeros(1, 1, 28, 28)

    out = cnn(tens)
    print(out.shape)

if __name__ == "__main__":
    main()