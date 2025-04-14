from torch import nn
import torch
from simplellm.gpt.gptbase import *
from simplellm.utils import IterableModule


class CausalGPT(IterableModule, nn.Module):
    def __init__(self, vocab_size, dmodel, num_heads, dim_feedforward = 0, norm_eps = 1e-5, dropout_prob = 0.1, ctx_size = 2048, device = "cuda", n_layers = 4):
        super().__init__()
        self.tokens_embed = nn.Embedding(vocab_size, dmodel).to(device)
        self.positions_embed = nn.Embedding(ctx_size, dmodel).to(device)
        self.drop = nn.Dropout(dropout_prob)
        self.register_buffer("position_ids", torch.arange(ctx_size).to(device), persistent=False)
        self.h = nn.ModuleList([
                    GPTBlock(
                        dmodel=dmodel,
                        num_heads=num_heads,
                        dim_feedforward=dim_feedforward,
                        norm_eps=norm_eps,
                        dropout_prob=dropout_prob,
                        ctx_size=ctx_size,
                        device=device
                    ) for _ in range(n_layers)
                ])
    
    def forward(self, x):
        _, sz = x.shape
        positions = self.position_ids[None, : sz]
        word_embeddings = self.tokens_embed(x)
        pos_embeddings = self.positions_embed(positions)
        h = self.drop(word_embeddings + pos_embeddings)
        for i, block in enumerate(self.h): 
            h = block(h)
        return h

class GPT(IterableModule, nn.Module):
    def __init__(self, mdl_type, vocab_size, dmodel, num_heads, dim_feedforward = 0, norm_eps = 1e-5, dropout_prob = 0.1, ctx_size = 2048, device = "cuda", n_layers = 4):
        super().__init__()
        self.transformer = mdl_type(vocab_size, dmodel, num_heads, dim_feedforward = dim_feedforward, norm_eps = norm_eps, dropout_prob = dropout_prob, ctx_size = ctx_size, device = device, n_layers = n_layers)
        self.lm_head = nn.Linear(dmodel, vocab_size, bias=False, device=device)
    
    def forward(self, x):
        out = self.transformer(x)
        
        return self.lm_head(out)
           



class GPTStage(nn.Module):
    def __init__(self, dmodel, num_heads, dim_feedforward = 0, norm_eps = 1e-5, dropout_prob = 0.1, ctx_size = 2048, device = "cuda", n_layers = 4) -> None:
        super().__init__()
        self.transformers = nn.Sequential(
            *[
                GPTBlock(
                    dmodel=dmodel,
                    num_heads=num_heads,
                    dim_feedforward=dim_feedforward,
                    norm_eps=norm_eps,
                    dropout_prob=dropout_prob,
                    ctx_size=ctx_size,
                    device=device
                ) for _ in range(n_layers)
            ]
        )
    
    def forward(self, x):
        return self.transformers(x)

class GPTFirstStage(nn.Module):
    def __init__(self, vocab_size, dmodel, num_heads, n_layers = 4, multiple_of = 256, dropout_prob = 0.1, norm_eps = 1e-5, ffn_dim_multiplier = None, ctx_size = 2048, padding_idx = None, device = "cuda", share_weights = False, de_embed = True) -> None:
        super().__init__()
        self._embedding = GPTEmbedding(vocab_size,dmodel,ctx_size,padding_idx = padding_idx,device = device)
        
        if de_embed:
            self.lm_head = nn.Linear(dmodel, vocab_size, bias=False,device=device)
            if share_weights:
                self._embedding.weight = self.lm_head.weight
        self.n_layers = n_layers
        if self.n_layers > 0:
            self.layers = nn.Sequential(
            *[
                GPTBlock(
                    dmodel=dmodel,
                    num_heads=num_heads,
                    dim_feedforward=dim_feedforward,
                    norm_eps=norm_eps,
                    dropout_prob=dropout_prob,
                    ctx_size=ctx_size,
                    device=device
                ) for _ in range(n_layers)
            ]
        )

    def embed(self, x):
        x = self._embedding(x)
        if self.n_layers == 0:
            return x
        h = self.layers(x)
        return h
    def forward_end(self, x):
        return  self.lm_head(x)


