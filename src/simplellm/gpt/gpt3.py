from torch import nn
import torch
from simplellm.gpt.gptbase import *

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


