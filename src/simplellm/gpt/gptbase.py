from typing import Literal
from torch import nn
import torch

class Conv1D(nn.Module):
    def __init__(self, nf, nx, device = "cuda"):
        super().__init__()
        self.nf = nf
        self.nx = nx
        self.weight = nn.Parameter(torch.empty(nx, nf).to(device))
        self.bias = nn.Parameter(torch.zeros(nf).to(device))
        nn.init.normal_(self.weight, std=0.02)

    def __repr__(self) -> str:
        return "Conv1D(nf={nf}, nx={nx})".format(**self.__dict__)

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(size_out)
        return x
    
class Attention(nn.Module):
    def __init__(self, dmodel, ctx_size, num_heads, dropout_prob = 0.1, device = "cuda"):
        super().__init__()
        
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(ctx_size, ctx_size)).view(1, 1, ctx_size, ctx_size).to(device),
            persistent=False,
        )
        self.num_heads = num_heads
        self.split_size = dmodel
        self.c_attn = Conv1D(dmodel * 3, dmodel, device)
        self.c_proj = Conv1D(dmodel, dmodel, device)
        self.attn_dropout = nn.Dropout(dropout_prob)
        self.resid_dropout = nn.Dropout(dropout_prob)
        

    def _attn(self, q, k, v):
        w = torch.matmul(q, k)
        
        b = self.bias[:, :, : w.size(-2), : w.size(-1)]
        w = w * b + -1e4 * (1 - b)


        w = nn.functional.softmax(w, dim=-1)
        w = self.attn_dropout(w)

        
        return torch.matmul(w, v)

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)
        if k:
            return x.permute(0, 2, 3, 1)
        else:
            return x.permute(0, 2, 1, 3)

    def forward(self, x):
        x = self.c_attn(x)
        print(x.size(), self.split_size)
        query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)

        a = self._attn(query, key, value)

        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)

        return a 

class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class MLP(nn.Module):
    def __init__(self, dmodel, dim_feedforward, dropout_prob = 0.1, device = "cuda"):
        super().__init__()
        self.c_fc = Conv1D(dim_feedforward, dmodel, device)
        self.c_proj = Conv1D(dmodel, dim_feedforward, device)
        self.act = NewGELU().to(device)
        self.dropout = nn.Dropout(dropout_prob).to(device)

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)



class GPTBlock(nn.Module):
    def __init__(self, dmodel, num_heads, dim_feedforward = 0, norm_eps = 1e-5, dropout_prob = 0.1, ctx_size = 2048, device = "cuda") -> None:
        super().__init__()
        if dim_feedforward == 0:
            dim_feedforward = dmodel * 4
        self.attn = Attention(dmodel, ctx_size, num_heads, dropout_prob, device)
        self.norm1 = nn.LayerNorm(dmodel, eps=norm_eps).to(device)
        self.mlp = MLP(dmodel, dim_feedforward, dropout_prob, device)
        self.norm2 = nn.LayerNorm(dmodel, eps=norm_eps).to(device)
        
    def forward(self, x, mask=None):
        x_ = self.attn(x)
        x_ = self.norm1(x + x_)
        m = self.mlp(x_)
        return self.norm2(x_ + m)

    

class GPTEmbedding(nn.Module):
    def __init__(self, vocab_size, dmodel, ctx_size = 2048, dropout_prob = 0.1, padding_idx = None, device = "cuda") -> None:
        super().__init__()
        self.word_embedding = nn.Embedding(vocab_size, dmodel, padding_idx=padding_idx).to(device)
        self.pos_embedding = nn.Embedding(ctx_size, dmodel).to(device)
        self.drop = nn.Dropout(dropout_prob)
    
    def forward(self, x, positions = None):
       
        _, sz = x.shape
        if positions == None:
            positions = torch.arange(0, sz, device=x.device, dtype=torch.long).unsqueeze(0)
        word_embeddings = self.word_embedding(x)
        pos_embeddings = self.pos_embedding(positions)[None, ...]
        return self.drop(word_embeddings + pos_embeddings)



    