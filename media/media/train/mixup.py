from lightning.pytorch.callbacks import Callback
import numpy as np
import torch

class MixupCallback(Callback):
    def __init__(self, alpha=0.2):
        super().__init__()
        self.alpha = alpha

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        inputs, targets = batch
        lam = np.random.beta(self.alpha, self.alpha) if self.alpha > 0 else 1
        index = torch.randperm(inputs.size(0)).to(inputs.device)
        
        mixed_inputs = lam * inputs + (1 - lam) * inputs[index, :]
        targets_a, targets_b = targets, targets[index]
        batch[0], batch[1] = mixed_inputs, (targets_a, targets_b, lam)
        return batch