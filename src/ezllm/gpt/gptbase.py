from typing import Literal
from torch import nn
import torch

class GPTBlock(nn.Module):
    def __init__(self, dmodel, num_heads, dim_feedforward = 0, norm_eps = 1e-5, dropout_prob = 0.1, ctx_size = 2048, device = "cuda") -> None:
        super().__init__()
        if dim_feedforward == 0:
            dim_feedforward = dmodel * 4
        self.multihead_attn = nn.MultiheadAttention(embed_dim = dmodel, num_heads = num_heads, bias=True, dim_feedforward = dim_feedforward, batch_first = True).to(device)
        self.norm1 = nn.LayerNorm(dmodel, eps=norm_eps).to(device)
        self.norm2 = nn.LayerNorm(dmodel, eps=norm_eps).to(device)
        self.register_buffer("masked_attention",(1 - torch.tril(torch.ones(ctx_size, ctx_size))).to(device = device, dtype=torch.bool))
        self.mlp = nn.Sequential(
            nn.Linear(dmodel, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, dmodel),
            nn.DropOut(dropout_prob)
        ).to(device)
    def forward(self, x):
        _, tkns, _ = x.size()
        x_ = self.norm1(x)
        x_, _ = self.multihead_attn(x_, x_, x_, attn_mask=self.masked_attention[:tkns,:tkns])
        x = x_ + x
        x_ = self.norm2(x)
        
        x = x + self.mlp(x_)
        return x

    

class GPTEmbedding(nn.Module):
    def __init__(self, vocab_size, dmodel, ctx_size = 2048, device = "cuda") -> None:
        super().__init__()
        self.word_embedding = nn.Embedding(vocab_size, dmodel, padding_idx=None, max_norm=None,  norm_type=2, scale_grad_by_freq=False, sparse=False).to(device)
        self.pos_embedding = nn.Embedding(ctx_size, dmodel).to(device)
    
    def forward(self, x, positions = None):
       
        _, sz = x.shape
        if positions == None:
            positions = torch.arange(sz, device=x.device)
        word_embeddings = self.word_embedding(x)
        pos_embeddings = self.pos_embedding(positions)[None, ...]
        return word_embeddings + pos_embeddings

class GPTClassification(nn.Module):
    def __init__(self, vocab_size, dmodel, norm_eps=1e-5, type: Literal["cross_entropy", "seq_2_seq"] = "cross_entropy", device = "cuda") -> None:
        super().__init__()
        self.type = type
        self.norm1 = nn.LayerNorm(dmodel, eps=norm_eps).to(device)
        
        self.sfmx = nn.AdaptiveLogSoftmaxWithLoss(dmodel, vocab_size, [100, 1000, 10000])

    def forward(self, x, targets):
        if self.type == "cross_entropy":
            return nn.functional.cross_entropy(x.view(-1, x.size(-1)), targets.view(-1))
        elif self.type == "seq_2_seq":
            # from : https://github.com/DS3Lab/DT-FM
            x = self.norm1(x)
            
            shifted_x = x[..., :-1, :].contiguous()
            shifted_targets = targets[..., 1:].contiguous()
            return self.sfmx(shifted_x.view(-1, self.norm1.in_features), shifted_targets.view(-1)).loss
        else:
            raise NotImplemented(f"Not a valid method ${self.type}")

    