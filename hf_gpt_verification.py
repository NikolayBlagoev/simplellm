from transformers import OpenAIGPTConfig
from transformers.models.openai.modeling_openai import OpenAIGPTModel, OpenAIGPTLMHeadModel
from simplellm.tokenizers import SPTokenizer, GPTTokenizer
from simplellm.gpt import GPTEmbedding, GPTBlock, GPT, CausalGPT
from simplellm.losses import causalLLMLoss
import torch
dmodel = 128
num_heads = 16
n_layers = 4
seq_l = 128
batch_size = 4
device = "cpu"

tkns = GPTTokenizer()
print(tkns.encode("hi"))
config = OpenAIGPTConfig(tkns.vocab_size,n_positions=seq_l, n_embd=dmodel, n_layer=1, n_head=num_heads)
net = OpenAIGPTModel(config)
embed = net.tokens_embed
pos_embed = net.positions_embed

embed_mine = GPTEmbedding(tkns.vocab_size, dmodel, seq_l, device=device)
embed_mine.word_embedding.weight = net.tokens_embed.weight
embed_mine.pos_embedding.weight = net.positions_embed.weight
inp = torch.randint(100,(1,seq_l), dtype=torch.long, device = device).to(device)
ret1 = embed_mine(inp)

position_ids = net.position_ids[None, : seq_l]
inputs_embeds = embed(inp.to("cpu"))
position_embeds = pos_embed(position_ids.to("cpu"))
ret2 = embed_mine.drop(position_embeds + inputs_embeds)
print(ret1.shape)
print(ret2.shape)
print(ret1)
print(ret2)

config = OpenAIGPTConfig(tkns.vocab_size,n_positions=seq_l, n_embd=dmodel, n_layer=n_layers, n_head=num_heads)
net = OpenAIGPTLMHeadModel(config)
print(config)
net2 = GPT(CausalGPT, tkns.vocab_size, dmodel, num_heads, ctx_size=seq_l, device=device, n_layers=n_layers)
net2.load_state_dict(net.state_dict())
net2.eval()
net.eval()
inp = torch.randint(100,(1,seq_l), dtype=torch.long, device = device).to(device)
ret1 = net(inp).logits
ret2 = net2(inp)
print(ret1)
print(ret2)

assert torch.allclose(ret1,ret2)