import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualLayer(nn.Module):
    def __init__(self, dim, trainable=True):
        super().__init__()
        self.dim = dim
        self.dense1 = nn.Linear(dim, dim)
        self.batch1 = nn.BatchNorm1d(dim)
        self.dense2 = nn.Linear(dim, dim)
        self.batch2 = nn.BatchNorm1d(dim)

    def forward(self, x):
        assert self.dim == x.shape[-1], f"Expected input with {self.dim} features, got {x.shape[-1]}"
        y = self.dense1(x)
        # Transpose the last dimension to the second dimension for BatchNorm1d
        y = y.permute(0, 2, 1)
        y = self.batch1(y)
        # Transpose back
        y = y.permute(0, 2, 1)
        y = F.relu(y)
        y = self.dense2(y)
        # Transpose the last dimension to the second dimension for BatchNorm1d
        y = y.permute(0, 2, 1)
        y = self.batch2(y)
        # Transpose back
        y = y.permute(0, 2, 1)
        y += x
        return F.relu(y)

class ResidualNet(nn.Module):
    def __init__(self, num_layers, num_descriptors, training=True):
        super().__init__()
        self.layers = nn.ModuleList(
            [ResidualLayer(num_descriptors) for _ in range(num_layers)]
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x

def solve_lstsq(A, B):
    At = A.transpose(1, 2)
    Bt = B.transpose(1, 2)
    
    result = torch.linalg.lstsq(At, Bt)

    C = result.solution.transpose(1, 2)
    return C

def correspondenceMatrix(sigs, evecs_t):
    A, B = (torch.matmul(e, s) for (e, s) in zip(evecs_t, sigs))
    C = solve_lstsq(A, B)
    return C

def softCorrespondenceEnsemble(C, evecs_1_t, evecs_2):
    P = torch.matmul(torch.matmul(evecs_2, C), evecs_1_t)
    P = F.normalize(P, p=2, dim=1)
    P = P.transpose(1, 2)
    return P.pow(2)

def geodesicErrorEnsemble(P, dist_1, dist_2):
    dist_21 = torch.matmul(torch.matmul(P, dist_2), P.transpose(1, 2))
    unsupervised_loss = F.mse_loss(dist_21, dist_1)
    unsupervised_loss /= P.size(0) * P.size(1) * P.size(1)
    return unsupervised_loss

if __name__ == "__main__":
    mod = ResidualNet(5, 3000)
    print(mod)
