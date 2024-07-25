import os
import numpy as np
import torch

from .data_loading import generate_TFRecord
from .prediction import DfmPredictor
from .training import EnsembleTrainer

def process_directory(data_dir, checkpoint_dir, config, mesh_type):
    num_signatures = sum(config[f"number_{s}"] for s in ["hks", "wks", "gaussian"])
    num_vertices = int(0.95 * config["number_vertices"])
    number_epochs = config["deep_functional_maps"]["epochs"]
    lr, bs = [config["deep_functional_maps"][k] for k in ["learning_rate", "batch_size"]]

    dataset = generate_TFRecord(data_dir, num_vertices, mesh_type)

    trainer = EnsembleTrainer(dataset, num_signatures, lr, bs, checkpoint_dir, mesh_type)
    trainer.train(number_epochs)

    predictor = DfmPredictor(mesh_type, num_signatures, checkpoint_dir)
    dir_sigs = os.path.join(data_dir, "signatures")
    files = os.listdir(dir_sigs)
    for fn in files:
        fpath = os.path.join(dir_sigs, fn)
        x_raw = np.load(fpath)
        x_new = predictor(torch.tensor(x_raw))
        np.save(fpath, x_new.numpy())

    return

if __name__ == "__main__":
    pass
