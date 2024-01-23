import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(x)

        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, w=1, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = w * 16

        self.conv1 = nn.Conv2d(3, w * 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(w * 16)
        
        self.layer1 = self._make_layer(block, w * 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, w * 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, w * 64, num_blocks[2], stride=2)
        
        self.linear = nn.Linear(w * 64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = F.avg_pool2d(out, out.size()[3])

        out = out.view(out.size(0), -1)
        out = self.linear(out)
        
        return nn.functional.log_softmax(out, dim=1)
    

class ResNetSpec:
    def __init__(self, num_blocks):
        self._layer_spec        = list()
        self._perm_spec         = list()
        self._layer_spec_unique = list()
    
        modules = [
            "conv1.weight",
            "bn1.weight",
            "bn1.bias",
            "bn1.running_mean",
            "bn1.running_var"
        ]
        perms = [
            (0, -1),
            (0, -1),
            (0, -1),
            (0, -1),
            (0, -1)
        ]
        unique_modules = [
            "conv1",
            "bn1"
        ]

        self._layer_spec.append(modules)
        self._perm_spec.append(perms)
        self._layer_spec_unique.append(unique_modules)

        cnt = 1
        for idx, num in enumerate(num_blocks):
            for j in range(num):
                modules = [
                    f"layer{idx + 1}.{j}.conv1.weight",
                    f"layer{idx + 1}.{j}.bn1.weight",
                    f"layer{idx + 1}.{j}.bn1.bias",
                    f"layer{idx + 1}.{j}.bn1.running_mean",
                    f"layer{idx + 1}.{j}.bn1.running_var",

                    f"layer{idx + 1}.{j}.conv2.weight",
                    f"layer{idx + 1}.{j}.bn2.weight",
                    f"layer{idx + 1}.{j}.bn2.bias",
                    f"layer{idx + 1}.{j}.bn2.running_mean",
                    f"layer{idx + 1}.{j}.bn2.running_var"
                ]
                perms = [
                    (cnt, cnt - 1),
                    (cnt, -1),
                    (cnt, -1),
                    (cnt, -1),
                    (cnt, -1),

                    (cnt + 1, cnt),
                    (cnt + 1, -1),
                    (cnt + 1, -1),
                    (cnt + 1, -1),
                    (cnt + 1, -1),
                ]
                unique_modules = [
                    f"layer{idx}.{j}.conv1",
                    f"layer{idx}.{j}.bn1",
                    f"layer{idx}.{j}.conv2",
                    f"layer{idx}.{j}.bn2",
                ]

                self._layer_spec.append(modules)
                self._perm_spec.append(perms)
                self._layer_spec_unique.append(unique_modules)

                cnt += 1

        modules = [
            "linear.weight",
            "linear.bias",
        ]
        perms = [
            (-1, cnt),
            (-1, -1),
        ]
        unique_modules = [
            "linear"
        ]

        self._layer_spec.append(modules)
        self._perm_spec.append(perms)
        self._layer_spec_unique.append(unique_modules)

        for i, sp in enumerate(self._layer_spec):
            for k, s in enumerate(sp):
                print(s, self._perm_spec[i][k])

    @property
    def layer_spec(self):
        return self._layer_spec
    
    @property
    def layer_spec_unique(self):
        return self._layer_spec_unique
    
    @property
    def perm_spec(self):
        return self._perm_spec


def test_resnet():
    m = ResNet(BasicBlock, [3, 3, 3])
    x = torch.randn(2, 3, 32, 32)

    spec = ResNetSpec([3,3,3])
    
    out = m(x)

if __name__ == "__main__":
    test_resnet()