import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch import ones_like, exp, mean
def perplexityLoss(logits,target, attention_mask = None):
    shift_logits = logits[:, :-1, :].contiguous()
    shift_target = target[:, 1:].contiguous()
    shift_mask = None
    if attention_mask != None:
        shift_mask = attention_mask[:, 1:].sum(1)
    else:
        shift_mask = shift_target.shape[1]
    ls = F.cross_entropy(shift_logits.transpose(1, 2), shift_target,reduction="none")
    ls = ls.sum(1) / shift_mask
    ls = exp(ls)
    return mean(ls)

def perplexityLoss2(logits,target, attention_mask = None):
    shift_logits = logits[:, :-1, :].contiguous()
    shift_target = target[:, 1:].contiguous()
    shift_mask = None
    if attention_mask != None:
        shift_mask = attention_mask[:, 1:].to(shift_logits.dtype).contiguous()
    else:
        factor = shift_target.shape[1]
    shift_logits = F.log_softmax(shift_logits, dim=-1)
    shift_target = shift_logits.gather(dim=-1, index=shift_target.unsqueeze(-1)).squeeze(-1)
    if attention_mask != None:
        shift_target = shift_target * shift_mask
        negative_log_likelihood = -shift_target.sum(1) / shift_mask.sum(1)
    else:
        negative_log_likelihood = -shift_target.sum(1) / factor
    perplexities = exp(negative_log_likelihood)
    

    return mean(perplexities)
