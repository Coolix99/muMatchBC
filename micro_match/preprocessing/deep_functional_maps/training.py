import os
import datetime
import warnings
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import itertools

from .operations import ResidualNet, correspondenceMatrix, softCorrespondenceEnsemble, geodesicErrorEnsemble

warnings.filterwarnings("ignore", category=FutureWarning)

def pairwise(iterable):
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

class EnsembleTrainer:
    def __init__(self, dataset, num_sigs, lr, bs, checkpoint_dir, chkpt_name):
        self.func = ResidualNet(7, num_sigs)
        self.loss = torch.nn.MSELoss()
        self.optimiser = optim.Adam(self.func.parameters(), lr=lr)
        self.dataloader = DataLoader(dataset, batch_size=bs, shuffle=True)
        self.checkpoint_dir = checkpoint_dir
        self.chkpt_name = chkpt_name

    def train_step(self, x, y):
        e_x, et_x, s_x, g_x = x["evecs"], x["evecs_t"], x["sigs"], x["metric"]
        e_y, et_y, s_y, g_y = y["evecs"], y["evecs_t"], y["sigs"], y["metric"]
        sigs = [self.func(s) for s in (s_x, s_y)]
        C = correspondenceMatrix(sigs, [et_x, et_y])
        P = softCorrespondenceEnsemble(C, et_x, e_y)
        loss = geodesicErrorEnsemble(P, g_x, g_y)
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()
        return loss

    def train(self, number_epochs):
        best_loss = float('inf')
        for epoch in range(number_epochs):
            for x, y in pairwise(self.dataloader):
                try:
                    loss = self.train_step(x, y)
                except Exception as e:
                    print(f"Error caught: {e}")
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item()}")
            if loss < best_loss:
                best_loss = loss
                torch.save(self.func.state_dict(), os.path.join(self.checkpoint_dir, self.chkpt_name))

if __name__ == "__main__":
    pass
