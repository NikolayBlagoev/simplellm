import torch
import torch.nn.functional as F
from torch import nn
from typing import Optional, List, Tuple


# Based on the Mixtral implementation of MoE
class MoERouter(nn.Module):
    def __init__(self, expert_count: int, gate: Optional[nn.Module] = None, dim: Optional[int] = None, k = 3, jitter_noise = 0.1):
        if dim == None and gate == None:
            raise AttributeError("Both dim and gate cannot be None. At least one needs to be defined")
        super().__init__()
        
        self.expert_count = expert_count
        if gate == None:
            gate = nn.Linear(dim, expert_count, bias=False)
        self.gate = gate
        self.k = k
        self.jitter_noise = jitter_noise
        

    def forward(self, inputs: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        _, _, dim = inputs.shape
        if self.training and self.jitter_noise > 0:
            inputs *= torch.empty_like(inputs).uniform_(1 - self.jitter_noise, 1 + self.jitter_noise)
        inputs = inputs.view(-1, dim)
        gate_logits = self.gate(inputs)

        weights = F.softmax(gate_logits,dim = 1 , dtype=torch.float)
        weights, selected_experts = torch.topk(weights, self.k, dim = -1)
        weights /= weights.sum(dim=-1, keepdim = True)
        weights = weights.to(inputs.dtype)

        expert_mask = nn.functional.one_hot(selected_experts, num_classes=self.expert_count).permute(2, 1, 0)
        return (weights, expert_mask, gate_logits)

    def run_MoE(weights: torch.Tensor, expert_mask: torch.Tensor, experts: List[nn.Module], inputs: torch.Tensor):
        b, seql, dim = inputs.shape
        results = torch.zeros((b * seql, dim), dtype=inputs.dtype, device=inputs.device)
        for i, expert in enumerate(experts):
           
            idx, top_x = torch.where(expert_mask[i])
            current_state = inputs[None, top_x].reshape(-1, dim)
            current_hidden_states = expert(current_state) * weights[top_x, idx, None]
            results.index_add_(0, top_x, current_hidden_states.to(inputs.dtype))
        results = results.reshape(b, seql, dim)
        return results



class BMoERouter(nn.Module):
    def __init__(self, expert_count: int, gate: Optional[nn.Module] = None, sqldim: Optional[int] = None, k = 3, jitter_noise = 0.1, device = None):
        if sqldim == None and gate == None:
            raise AttributeError("Both sqldim and gate cannot be None. At least one needs to be defined")
        super().__init__()
        
        self.expert_count = expert_count
        if gate == None:
            gate = nn.Linear(sqldim, expert_count, bias=False, device=device)
        self.gate = gate
        self.k = k
        self.jitter_noise = jitter_noise
        

    def forward(self, inputs: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        b, _, _ = inputs.shape
        if self.training and self.jitter_noise > 0:
            inputs *= torch.empty_like(inputs).uniform_(1 - self.jitter_noise, 1 + self.jitter_noise)
        inputs = inputs.view(b, -1)
        gate_logits = self.gate(inputs)

        weights = F.softmax(gate_logits,dim = 1 , dtype=torch.float)
        weights, selected_experts = torch.topk(weights, self.k, dim = -1)
        weights /= weights.sum(dim=-1, keepdim = True)
        weights = weights.to(inputs.dtype)
        
        
        return (weights, selected_experts, gate_logits)
