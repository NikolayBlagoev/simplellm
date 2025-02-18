import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch import ones_like, exp, mean
def perplexityLoss(logits,target, attention_mask = None):
    shift_logits = logits[:, :-1, :].contiguous()
    shift_target = target[:, 1:].contiguous()
    shift_mask = None
    if attention_mask != None:
        shift_mask = attention_mask[:, 1:].to(shift_logits.dtype).contiguous()
    else:
        shift_mask = ones_like(shift_target).contiguous()

    ls = CrossEntropyLoss(reduction="none")
    ls = ls(shift_logits.transpose(1, 2), shift_target)
    ls = ls.sum(1) / shift_mask.sum(1)
    ls = exp(ls).tolist()
    return mean(ls)

def perplexityLoss2(logits,target, attention_mask = None):
    shift_logits = logits[:, :-1, :].contiguous()
    shift_target = target[:, 1:].contiguous()
    shift_mask = None
    if attention_mask != None:
        shift_mask = attention_mask[:, 1:].to(shift_logits.dtype).contiguous()
    else:
        shift_mask = ones_like(shift_target).contiguous()
    shift_logits = F.log_softmax(shift_logits, dim=-1)
    shift_target = shift_logits.gather(dim=-1, index=shift_target.unsqueeze(-1)).squeeze(-1)

    shift_target = shift_target * shift_mask
    negative_log_likelihood = -shift_target.sum(-1) / shift_mask.sum(-1)

    perplexities = exp(negative_log_likelihood)
    perplexities = perplexities.tolist()

    return mean(perplexities)
