import torch

def check_weights(model1, model2):
    for (p1, p2) in zip(model1.parameters(), model2.parameters()):
        if not torch.equal(p1, p2):
            return False
    return True

