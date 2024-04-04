from llamabase import *

import torch

from torch import nn
class LLama(nn.Module):
    def __init__(self, vocab_size, dmodel = 4096, num_heads = 32, multiple_of = 256, norm_eps = 1e-5, dropout_prob = 1e2, ctx_size = 2048, device = "cuda", n_layers = 4, ffn_dim_multiplier = None) -> None:
        super().__init__()
        self.embedding = LLamaEmbedding(vocab_size,dmodel)
        self.freqs_cis = precompute_freqs_cis(dmodel // num_heads, ctx_size * 2)
        self.transformers = []
        for _ in range(n_layers):
            self.transformers.append(TransformerBlock(
                    dmodel=dmodel,
                    num_heads=num_heads,
                    freq_cis=self.freqs_cis,
                    multiple_of=multiple_of,
                    norm_eps=norm_eps,
                    ffn_dim_multiplier=ffn_dim_multiplier, 
                    device = device
                ))
        self.norm = RMSNorm(dmodel, eps=norm_eps)
        self.ln = nn.Linear(dmodel, vocab_size, bias=False)
    def forward(self, x, start_p):
        _, seq_l = x.shape
        h = self.embedding(x)
        # print(h.type())
        mask = None
        if seq_l > 1:
            mask = torch.full(
                (seq_l, seq_l), float("-inf"), device=x.device
            )

            mask = torch.triu(mask, diagonal=1)

            # When performing key-value caching, we compute the attention scores
            # only for the new sequence. Thus, the matrix of scores is of size
            # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
            # j > cache_len + i, since row i corresponds to token cache_len + i.
            mask = torch.hstack([
                torch.zeros((seq_l, start_p), device=x.device),
                mask
            ]).type_as(h)
        for layer in self.transformers:
            h = layer(h, start_p, seq_l, mask)
            # print(len(bytes(pickle.dumps(h))))
            # print(h.type())
        h = self.norm(h)
        output = self.ln(h).float()
        return output



class LLamaStage(nn.Module):
    def __init__(self, dmodel, num_heads, n_layers = 4, multiple_of = 256, norm_eps = 1e-5, ffn_dim_multiplier = None, ctx_size = 2048, device = "cuda") -> None:
        super().__init__()
        self.transformers = []
        self.freqs_cis = precompute_freqs_cis(dmodel // num_heads, ctx_size * 2)
        for _ in range(n_layers):
            self.transformers.append(TransformerBlock(
                    dmodel=dmodel,
                    num_heads=num_heads,
                    freq_cis=self.freqs_cis,
                    multiple_of=multiple_of,
                    norm_eps=norm_eps,
                    ffn_dim_multiplier=ffn_dim_multiplier, 
                    device = device
                ))
        
    def forward(self, x, start_p, seq_l):
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
        for t in self.transformers:
            x = t(x, start_p, seq_l,mask)
        return x

class LLamaFirstStage(nn.Module):
    def __init__(self, vocab_size, dmodel, num_heads, n_layers = 4, multiple_of = 256, norm_eps = 1e-5, ffn_dim_multiplier = None, ctx_size = 2048, device = "cuda") -> None:
        super().__init__()
        self.embedding = LLamaEmbedding(vocab_size,dmodel)
        self.freqs_cis = precompute_freqs_cis(dmodel // num_heads, ctx_size * 2)
        self.transformers = []
        for _ in range(n_layers):
            self.transformers.append(TransformerBlock(
                    dmodel=dmodel,
                    num_heads=num_heads,
                    freq_cis=self.freqs_cis,
                    multiple_of=multiple_of,
                    norm_eps=norm_eps,
                    ffn_dim_multiplier=ffn_dim_multiplier, 
                    device = device
                ))
        
    def forward(self, x, start_p):
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
        
        for t in self.transformers:
            x = t(x, start_p, seq_l,mask)
        return x, seq_l