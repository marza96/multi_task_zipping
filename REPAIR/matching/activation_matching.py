import torch
import tqdm

from .matching_utils import solve_lap, apply_permutation, permmat_to_perm


class ActivationMatching:
    def __init__(self, loader, ret_perm=False, debug=False, epochs=1, device="cpu") -> None:
        self.debug    = debug
        self.loader   = loader
        self.epochs   = epochs
        self.ret_perm = ret_perm
        self.device   = device

    def __call__(self, layer_indices, net0, net1):
        permutations = list()

        for _, layer_idx in enumerate(layer_indices[:-1]):
            obj = self.corr_matrix(
                net0.subnet(net0, layer_i=layer_idx), 
                net1.subnet(net1, layer_i=layer_idx), 
            )

            permutations.append(solve_lap(obj))

        if self.ret_perm is True:
            return permutations
        
        out_shape = net0.layers[-1].weight.shape[0]
        permutations += [permmat_to_perm(torch.eye(out_shape))]
        
        net1 = apply_permutation(layer_indices, net1, permutations)

        return net0, net1
    
    def corr_matrix(self, net0, net1):
        n = self.epochs * len(self.loader)
        mean0 = mean1 = std0 = std1 = outer = None
        with torch.no_grad():
            net0.eval()
            net1.eval()

            for _ in range(self.epochs):
                for i, (images, _) in enumerate(tqdm.tqdm(self.loader)):
                    img_t = images.float().to(self.device)

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
