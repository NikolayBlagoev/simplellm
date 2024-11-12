import torch.nn.functional as F
def causalLLMLoss(x, target, vocab_size):
    
    shift_logits = x.float()[..., :-1, :].contiguous()
    shift_labels = target[..., 1:].contiguous()

    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    
    loss = F.cross_entropy(shift_logits, shift_labels)
    return loss