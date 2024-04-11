from torch import nn
import torch
from simplellm.gpt.gptbase import *

class GPTStage(object):
    def __init__(self, dmodel, num_heads, dim_feedforward = 0, norm_eps = 1e-5, dropout_prob = 1e2, ctx_size = 2048, device = "cuda", n_layers = 4) -> None:
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

class GPTFirstStage(object):
    def __init__(self, vocab_size, dmodel, num_heads, dim_feedforward = 0, norm_eps = 1e-5, dropout_prob = 1e2, ctx_size = 2048, device = "cuda", n_layers = 4) -> None:
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
        self.embedding = GPTEmbedding(vocab_size,dmodel,ctx_size,device)
    
    def forward(self, x):

        return self.transformers(self.embedding(x))

class GPT3(object):
    def __init__(self, vocab_size, dmodel, num_heads, dim_feedforward = 0, norm_eps = 1e-5, dropout_prob = 1e2, ctx_size = 2048, device = "cuda", n_layers = 96, loss_type: Literal["cross_entropy", "seq_2_seq"] = "cross_entropy") -> None:
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
        self.embedding = GPTEmbedding(vocab_size,dmodel,ctx_size,device)
        self.classification = GPTClassification(vocab_size,dmodel,norm_eps,loss_type,device)
    
    def forward(self, x):

        return self.transformers(self.embedding(x))
    def loss(self, x, targets):
        return self.classification(x,targets)
