from typing import Tuple, List
import torch


def zloss(gate_logits: List[torch.Tensor], num_experts: int, top_k: int):
    gate_logits = torch.cat(gate_logits,dim = 0).to(gate_logits[0].device)
    routing_weights = torch.nn.functional.softmax(gate_logits, dim=-1)
    _, top_k = torch.topk(gate_logits, top_k, dim=-1)
    expert_mask = torch.nn.functional.one_hot(top_k, num_experts)
    tokens_per_expert = torch.mean(expert_mask.float(),dim = 0)
    router_prob = torch.mean(routing_weights, dim = 0)
    return torch.sum(tokens_per_expert * router_prob.unsqueeze(0)) * num_experts