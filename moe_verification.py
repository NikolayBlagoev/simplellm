from simplellm.MoE.router import BMoERouter
import torch

net = BMoERouter(4,sqldim=50,k=1)

inp = torch.randn((4,5,10))
weights, selected_experts, logits = net(inp)
print(weights)

print(selected_experts)

batch_idx = torch.where(selected_experts == 3)
print(batch_idx)
# print(nth_expert)