import torch.nn.functional as F
import torch

def kl_loss(x: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None):
    r = target.float() - x.float()
    if mask is not None:
        r = r * action_mask
    return r.exp() - r - 1



