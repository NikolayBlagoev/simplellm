import torch.nn.functional as F
import torch


def grpo_loss(log_probs: torch.Tensor, original_log_probs: torch.Tensor, rewards: torch.Tensor,  msk: torch.Tensor = None, kl = 0, eps = 0.26, kl_weight = 0.01, reduction = "mean"):
    r = (log_probs - original_log_probs).exp()
    loss = -torch.min(r * rewards, r.clamp(1 - eps, 1 + eps) * rewards)
    loss += kl_weight * kl
    if msk is None:
        loss = msk.mean(dim=-1)
    else:
        loss = ((loss * msk).sum(axis=-1) / msk.sum(axis=-1))
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    