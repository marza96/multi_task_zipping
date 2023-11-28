from typing import Any
import torch
import tqdm

from .utils import solve_lap


class ActivationMatching:
    def __init__(self, loader, debug=False, epochs=1) -> None:
        self.debug  = debug
        self.loader = loader
        self.epochs = epochs

    def __call__(self, layer_indices, net0, net1, device="cuda"):
        permutations = list()

        for _, layer_idx in enumerate(layer_indices):
            obj = corr_matrix(
                net0.subnet(net0, layer_i=layer_idx), 
                net1.subnet(net1, layer_i=layer_idx), 
                epochs=self.epochs, 
                loader=self.loader, 
                device=device
            )

            permutations.append(solve_lap(obj))

        return permutations
    
    def corr_matrix(self, net0, net1, device=None):
        n = self.epochs * len(self.loader)
        mean0 = mean1 = std0 = std1 = outer = None
        with torch.no_grad():
            net0.eval()
            net1.eval()

            for _ in range(self.epochs):
                for i, (images, _) in enumerate(tqdm(self.loader)):
                    img_t = images.float().to(device)
                    out0 = net0(img_t)
                    out0 = out0.reshape(out0.shape[0], out0.shape[1], -1).permute(0, 2, 1)
                    out0 = out0.reshape(-1, out0.shape[2]).float()

                    out1 = net1(img_t)
                    out1 = out1.reshape(out1.shape[0], out1.shape[1], -1).permute(0, 2, 1)
                    out1 = out1.reshape(-1, out1.shape[2]).float()

                    mean0_b = out0.mean(dim=0)
                    mean1_b = out1.mean(dim=0)
                    std0_b = out0.std(dim=0)
                    std1_b = out1.std(dim=0)
                    outer_b = (out0.T @ out1) / out0.shape[0]

                    if i == 0:
                        mean0 = torch.zeros_like(mean0_b)
                        mean1 = torch.zeros_like(mean1_b)
                        std0 = torch.zeros_like(std0_b)
                        std1 = torch.zeros_like(std1_b)
                        outer = torch.zeros_like(outer_b)

                    mean0 += mean0_b / n
                    mean1 += mean1_b / n
                    std0 += std0_b / n
                    std1 += std1_b / n
                    outer += outer_b / n

        cov = outer - torch.outer(mean0, mean1)
        corr = cov / (torch.outer(std0, std1) + 1e-4)

        return corr


def act_similarity(net0, net1, epochs=1, loader=None, device=None):
    n = epochs * len(loader)
    F = None

    with torch.no_grad():
        net0.eval()
        net1.eval()

        for _ in range(epochs):
            for i, (images, _) in enumerate(tqdm(loader)):
                img_t = images.float().to(device)
                out0 = net0(img_t)
                out0 = out0.reshape(out0.shape[0], out0.shape[1], -1).permute(0, 2, 1)
                out0 = out0.reshape(-1, out0.shape[2]).float()

                out1 = net1(img_t)
                out1 = out1.reshape(out1.shape[0], out1.shape[1], -1).permute(0, 2, 1)
                out1 = out1.reshape(-1, out1.shape[2]).float()

                prod = out0.T @ out1

                if i == 0:
                    F = torch.zeros(prod.shape).to(device)
                
                F += prod

    return F


def corr_matrix(net0, net1, epochs=1, loader=None, device=None):
    n = epochs * len(loader)
    mean0 = mean1 = std0 = std1 = outer = None
    with torch.no_grad():
        net0.eval()
        net1.eval()

        for _ in range(epochs):
            for i, (images, _) in enumerate(tqdm(loader)):
                img_t = images.float().to(device)
                out0 = net0(img_t)
                out0 = out0.reshape(out0.shape[0], out0.shape[1], -1).permute(0, 2, 1)
                out0 = out0.reshape(-1, out0.shape[2]).float()

                out1 = net1(img_t)
                out1 = out1.reshape(out1.shape[0], out1.shape[1], -1).permute(0, 2, 1)
                out1 = out1.reshape(-1, out1.shape[2]).float()

                mean0_b = out0.mean(dim=0)
                mean1_b = out1.mean(dim=0)
                std0_b = out0.std(dim=0)
                std1_b = out1.std(dim=0)
                outer_b = (out0.T @ out1) / out0.shape[0]

                if i == 0:
                    mean0 = torch.zeros_like(mean0_b)
                    mean1 = torch.zeros_like(mean1_b)
                    std0 = torch.zeros_like(std0_b)
                    std1 = torch.zeros_like(std1_b)
                    outer = torch.zeros_like(outer_b)

                mean0 += mean0_b / n
                mean1 += mean1_b / n
                std0 += std0_b / n
                std1 += std1_b / n
                outer += outer_b / n

    cov = outer - torch.outer(mean0, mean1)
    corr = cov / (torch.outer(std0, std1) + 1e-4)

    return corr


def activation_matching(layer_indices, net0, net1, loader, epochs=1, device="cuda", debug=False):
    permutations = list()

    for i, layer_idx in enumerate(layer_indices):
        obj = corr_matrix(
            net0.subnet(net0, layer_i=layer_idx), 
            net1.subnet(net1, layer_i=layer_idx), 
            epochs=epochs, 
            loader=loader, 
            device=device
        )

        permutations.append(solve_lap(obj))

    return permutations