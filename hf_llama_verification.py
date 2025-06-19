from transformers.models.llama.modeling_llama import LlamaMLP, LlamaAttention, LlamaRotaryEmbedding
from transformers import LlamaConfig, LlamaForCausalLM
from simplellm.llama.llamabase import FeedForward, Attention, RoPE
from simplellm.llama import CausalLLama, LLama
from simplellm.tokenizers import SPTokenizer, GPTTokenizer
from simplellm.losses import causalLLMLoss
from simplellm.dataloaders import OpenWebText

from copy import deepcopy
import torch
dmodel = 32
num_heads = 4
n_layers = 2
seq_l = 32
batch_size = 4
device = "cuda"


tkns = SPTokenizer()
config = LlamaConfig(tkns.vocab_size,hidden_size=dmodel,intermediate_size=dmodel*4,num_hidden_layers=n_layers,num_attention_heads=num_heads,max_position_embeddings=seq_l,device_map=device)
config._attn_implementation = "sdpa"
print(config)
inp = torch.rand((1,seq_l,dmodel))
net = LlamaMLP(config)
net2 = FeedForward(dmodel,dmodel*4,256,None,device="cpu")

net2.load_state_dict(net.state_dict())
ret1 = net(inp)
ret2 = net2(inp)

assert torch.allclose(ret1,ret2)

rotary_emb = LlamaRotaryEmbedding(config=config)
ours_rot = RoPE(dmodel // num_heads, device="cpu")
position_ids = torch.arange(0, seq_l).unsqueeze(0)
position_embeddings = rotary_emb(inp, position_ids)
net = LlamaAttention(config,0)
net2 = Attention(dmodel,num_heads,seq_l,device="cpu")
net2.load_state_dict(net.state_dict())
ret1, _ = net(inp,attention_mask = None, position_embeddings = position_embeddings)
position_embeddings = ours_rot(inp, 1, seq_l)
ret2 = net2(inp,position_embedding=position_embeddings)
print(ret1)
print(ret2)
assert torch.allclose(ret1,ret2)

torch.manual_seed(10)
tokenizer = SPTokenizer()

net = LlamaForCausalLM(config)
print(net)
torch.manual_seed(10)
net2 = LLama(CausalLLama, tokenizer.vocab_size, dmodel=dmodel, num_heads=num_heads, dropout_prob=0, ctx_size=seq_l, device="cpu", n_layers=n_layers)
print(net2)
# print(net2.model.layers[2])
# print(net.model.layers[2])
# print(net2.state_dict())
# print(net.state_dict())



net2.load_state_dict(deepcopy(net.state_dict()), strict=True)

# inp = torch.randint(100,(1,seq_l), dtype=torch.long, device = device).to("cpu")
# inpt = inp.detach().clone()
# target = torch.randint(100,(1,seq_l), dtype=torch.long, device = device).to("cpu")
# ret = net(inp,labels=target)
# print(ret)

# ret2 = net2(inpt)
# print(causalLLMLoss(ret2,target,tkns.vocab_size))
# print(ret2)
# assert torch.allclose(ret.logits,ret2)

inp = torch.randint(100,(4,seq_l), dtype=torch.long, device = device).to("cpu")
# inp.requires_grad = True
inpt = inp.detach().clone()
# inpt.requires_grad = True
target = torch.randint(100,(4,seq_l), dtype=torch.long, device = device).to("cpu")
ret = net(inp,labels=target)
ret.loss.backward()
print(ret)
ret2 = net2(inpt)
l2 = causalLLMLoss(ret2,target,tkns.vocab_size)
l2.backward()
print(ret2)
print(l2)
assert torch.allclose(ret.logits,ret2)
# print(net2.model.layers[1].self_attn.q_proj.weight.grad)
# print(net.model.layers[1].self_attn.q_proj.weight.grad)
print(net2.lm_head.weight.grad)
print(net.lm_head.weight.grad)
assert torch.allclose(net2.lm_head.weight.grad,net.lm_head.weight.grad)
assert torch.allclose(net2.model.layers[1].self_attn.q_proj.weight.grad,net.model.layers[1].self_attn.q_proj.weight.grad)
# print(config)