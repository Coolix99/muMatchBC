import os
import numpy as np
import torch

from .operations import ResidualNet

class DfmPredictor:
    def __init__(self, chkpt_name, num_signatures, checkpoint_dir):
        self.func = ResidualNet(7, num_signatures)
        weight_path = os.path.join(checkpoint_dir, chkpt_name)
        self.func.load_state_dict(torch.load(weight_path))
        self.func.eval()

    def __call__(self, raw_signatures):
        raw_signatures = torch.tensor(raw_signatures[np.newaxis]).float()
        with torch.no_grad():
            improved_signatures = self.func(raw_signatures)[0]
        return improved_signatures

if __name__ == "__main__":
    pass
