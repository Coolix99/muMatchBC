import os
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import itertools

warnings.filterwarnings("ignore", category=FutureWarning)

class SignaturesDataset(Dataset):
    def __init__(self, dir_in, N):
        self.dir_in = dir_in
        self.N = N
        self.names = [
            os.path.splitext(f)[0] for f in os.listdir(os.path.join(dir_in, "signatures"))
        ]

    def pruneIndices(self, number, target):
        return np.linspace(0, number, target, endpoint=False).astype(np.int)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        fn = self.names[idx]
        s = np.load(os.path.join(self.dir_in, "signatures", fn + ".npy")).astype(np.float32)
        eigen = np.load(os.path.join(self.dir_in, "eigen", fn + ".npz"))
        e, e_t = [eigen[k].astype(np.float32) for k in ["evecs", "evecs_t"]]
        g = np.load(os.path.join(self.dir_in, "geodesic_matrices", fn + ".npy")).astype(np.float32)
        i = self.pruneIndices(e.shape[0], self.N)
        g /= np.mean(g)
        
        return {
            "evecs": torch.tensor(e[i]),
            "evecs_t": torch.tensor(e_t[:, i]),
            "metric": torch.tensor(g[i][:, i]),
            "sigs": torch.tensor(s[i]),
            "N_eigs": e.shape[-1],
            "N_vert": self.N,
            "N_sigs": s.shape[-1]
        }

def generate_TFRecord(dir_in, N, output_name):
    dataset = SignaturesDataset(dir_in, N)
    return dataset

if __name__ == "__main__":
    pass
