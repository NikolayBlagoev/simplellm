import torch.nn.functional as F
def causalLLMLoss(x, target, vocab_size, pad_idx = -100):
    
    x = x.float()
    target = target.to(x.device)
    target = F.pad(target, (0,1), value=pad_idx)
    shift_labels = target[..., 1:].contiguous()

    x = x.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    
    loss = F.cross_entropy(x, shift_labels,ignore_index=pad_idx,reduction="mean")
    return loss

