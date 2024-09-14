from simplellm.llama.llamabase import *

import torch

from torch import nn
class LLama(nn.Module):
    def __init__(self, vocab_size, dmodel = 4096, num_heads = 32, multiple_of = 256, norm_eps = 1e-5, dropout_prob = 1e2, ctx_size = 2048, padding_idx = None, device = "cuda", n_layers = 4, ffn_dim_multiplier = None) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, dmodel, padding_idx = padding_idx,device=device)
        self.freqs_cis = precompute_freqs_cis(dmodel // num_heads, ctx_size * 2).to(device)
        self.transformers = nn.Sequential(
            *[
                TransformerBlock(
                    dmodel=dmodel,
                    num_heads=num_heads,
                    freq_cis=self.freqs_cis,
                    multiple_of=multiple_of,
                    norm_eps=norm_eps,
                    ffn_dim_multiplier=ffn_dim_multiplier, 
                    idx = i,
                    device = device
                ) for i in range(n_layers)
            ])
        self.norm = RMSNorm(dmodel, eps=norm_eps,device=device)
        # self.ln = nn.Linear(dmodel, vocab_size, bias=False,device=device)
    def forward(self, x):
        _, seq_l = x.shape
        h = self.tok_embeddings(x)
        
        h = self.transformers(h)
        h = self.norm(h)
        # output = self.ln(h).float()
        return h


class SkipSeq(nn.Sequential):
    def forward(self, *inputs):
        x, start_p, mask, to_skip = inputs
        for module in self._modules.values():
            if module.idx in to_skip:
                continue
            x = module(x, start_p, mask)
        return x


class SwapSeq(nn.Sequential):
    def forward(self, *inputs):
        x, start_p, mask, to_skip = inputs
        for module in self._modules.values():
            if module.idx in to_skip:
                continue
            x = module(x, start_p, mask)
        return x

class SkipLLama(nn.Module):
    def __init__(self, vocab_size, dmodel = 4096, num_heads = 32, multiple_of = 256, norm_eps = 1e-5, dropout_prob = 1e2, ctx_size = 2048, padding_idx = None, device = "cuda", n_layers = 4, ffn_dim_multiplier = None) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, dmodel, padding_idx = padding_idx,device=device)
        self.freqs_cis = precompute_freqs_cis(dmodel // num_heads, ctx_size * 2).to(device)
        self.transformers = SkipSeq(
            *[
                TransformerBlock(
                    dmodel=dmodel,
                    num_heads=num_heads,
                    freq_cis=self.freqs_cis,
                    multiple_of=multiple_of,
                    norm_eps=norm_eps,
                    ffn_dim_multiplier=ffn_dim_multiplier, 
                    idx = i,
                    device = device
                ) for i in range(n_layers)
            ])
        self.norm = RMSNorm(dmodel, eps=norm_eps,device=device)
        # self.ln = nn.Linear(dmodel, vocab_size, bias=False,device=device)
    def forward(self, x, to_skip = []):
        _, seq_l = x.shape
        h = self.tok_embeddings(x)
        
        h = self.transformers(h, 0, None, to_skip)
        h = self.norm(h)
        # output = self.ln(h).float()
        return h


class SwapLLama(nn.Module):
    def __init__(self, vocab_size, dmodel = 4096, num_heads = 32, multiple_of = 256, norm_eps = 1e-5, dropout_prob = 1e2, ctx_size = 2048, padding_idx = None, device = "cuda", n_layers = 4, ffn_dim_multiplier = None) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, dmodel, padding_idx = padding_idx,device=device)
        self.freqs_cis = precompute_freqs_cis(dmodel // num_heads, ctx_size * 2).to(device)
        self.transformers = SwapSeq(
            *[
                TransformerBlock(
                    dmodel=dmodel,
                    num_heads=num_heads,
                    freq_cis=self.freqs_cis,
                    multiple_of=multiple_of,
                    norm_eps=norm_eps,
                    ffn_dim_multiplier=ffn_dim_multiplier, 
                    idx = i,
                    device = device
                ) for i in range(n_layers)
            ])
        self.norm = RMSNorm(dmodel, eps=norm_eps,device=device)
        # self.ln = nn.Linear(dmodel, vocab_size, bias=False,device=device)
    def forward(self, x, order = []):
        _, seq_l = x.shape
        h = self.tok_embeddings(x)
        
        h = self.transformers(h,0,None,order)
        h = self.norm(h)
        # output = self.ln(h).float()
        return h





class NextTokenLLama(nn.Module):
    def __init__(self, LLAMA_TYPE, vocab_size, dmodel = 4096, num_heads = 32, multiple_of = 256, norm_eps = 1e-5, dropout_prob = 1e2, ctx_size = 2048, padding_idx = None, device = "cuda", n_layers = 32, ffn_dim_multiplier = None) -> None:
        super().__init__()
        self.model = LLAMA_TYPE(vocab_size,dmodel,num_heads,multiple_of,norm_eps,dropout_prob,ctx_size,padding_idx,device,n_layers,ffn_dim_multiplier)
        self.lm_head = nn.Linear(dmodel, vocab_size, bias=False,device=device)
    def forward(self, x):
        return self.lm_head(self.model(x))

    



class LLamaStage(nn.Module):
    def __init__(self, dmodel, num_heads, n_layers = 4, multiple_of = 256, norm_eps = 1e-5, ffn_dim_multiplier = None, ctx_size = 2048, device = "cuda") -> None:
        super().__init__()
        self.transformers = []
        self.freqs_cis = precompute_freqs_cis(dmodel // num_heads, ctx_size * 2).to(device)
        self.transformers = nn.Sequential(
            *[
                TransformerBlock(
                    dmodel=dmodel,
                    num_heads=num_heads,
                    freq_cis=self.freqs_cis,
                    multiple_of=multiple_of,
                    norm_eps=norm_eps,
                    ffn_dim_multiplier=ffn_dim_multiplier, 
                    idx = i,
                    device = device
                ) for i in range(n_layers)
            ]
        
        )
        
    def forward(self, x, start_p = 0):
        mask = None
        
        _, seq_l, _ = x.shape
        if seq_l > 1:
            mask = torch.full(
                (seq_l, seq_l), float("-inf"), device=x.device
            )

            mask = torch.triu(mask, diagonal=1)
            mask = torch.hstack([
                torch.zeros((seq_l, start_p), device=x.device),
                mask
            ]).type_as(x)
        
        return self.transformers(x)

class LLamaFirstStage(nn.Module):
    def __init__(self, vocab_size, dmodel, num_heads, n_layers = 4, multiple_of = 256, norm_eps = 1e-5, ffn_dim_multiplier = None, ctx_size = 2048, padding_idx = None, device = "cuda") -> None:
        super().__init__()
        self.embedding = LLamaEmbedding(vocab_size,dmodel,padding_idx=padding_idx,device=device)
        self.freqs_cis = precompute_freqs_cis(dmodel // num_heads, ctx_size * 2).to(device)
        self.transformers = nn.Sequential(
            *[
                TransformerBlock(
                    dmodel=dmodel,
                    num_heads=num_heads,
                    freq_cis=self.freqs_cis,
                    multiple_of=multiple_of,
                    norm_eps=norm_eps,
                    ffn_dim_multiplier=ffn_dim_multiplier, 
                    idx = i,
                    device = device
                ) for i in range(n_layers)
            ]
        
        )
        
        
    def forward(self, x, start_p = 0):
        _, seq_l = x.shape
        x = self.embedding(x)
        mask = None
        if seq_l > 1:
            mask = torch.full(
                (seq_l, seq_l), float("-inf"), device=x.device
            )

            mask = torch.triu(mask, diagonal=1)
            mask = torch.hstack([
                torch.zeros((seq_l, start_p), device=x.device),
                mask
            ]).type_as(x)
        
        
        return self.transformers(x)