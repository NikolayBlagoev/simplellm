# Adapted form: https://github.com/meta-llama/llama/blob/main/llama/model.py
import math
from typing import Literal, Optional, Tuple
import torch
import torch.nn.functional as F
from torch import nn





class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, device = "cuda"):
        
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim,device=device))

    def _norm(self, x):
    
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        
        output = self._norm(x.float()).type_as(x)
        
        return output * self.weight

def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0):
    
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(seq_len, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)
def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    # print(xq_.shape, freqs_cis.shape)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class Attention(nn.Module):
   
    def __init__(self, dmodel, num_heads, freq_cis, num_kv_heads = None, device = "cuda"):
        
        super().__init__()
        self.n_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.head_dim = dmodel // num_heads
        self.num_heads = num_heads
        self.wq = nn.Linear(
            dmodel,
            num_heads * self.head_dim,
            bias=False,
            device=device
        )
        self.wk = nn.Linear(
            dmodel,
            self.n_kv_heads * self.head_dim,
            bias=False,
            device=device
        )
        self.wv = nn.Linear(
            dmodel,
            self.n_kv_heads * self.head_dim,
            bias=False,
            device=device
        )
        self.wo = nn.Linear(
            num_heads * self.head_dim,
            dmodel,
            bias=False,
            device=device
        )
        
        
        self.freq_cis = freq_cis
        
        

    def forward(
        self,
        x: torch.Tensor,
        start_p: int = 0,
        mask: Optional[torch.Tensor] = None
    ):
        
        bsz, seqlen, _ = x.shape
        
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.num_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=self.freq_cis[start_p: start_p+seqlen])
        keys = xk
        values = xv
        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2) # (bs, n_local_heads, cache_len + seqlen, head_dim)
        values = values.transpose(1, 2) # (bs, n_local_heads, cache_len + seqlen, head_dim)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
        device = "cuda"
    ):
        
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(
            dim, hidden_dim, bias=False,device=device)
        self.w2 = nn.Linear(
            hidden_dim, dim, bias=False,device=device)
        self.w3 = nn.Linear(
            dim, hidden_dim, bias=False,device=device)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, dmodel, num_heads, freq_cis, multiple_of = 256, norm_eps = 1e-5, ffn_dim_multiplier = None, device = "cuda"):
        super().__init__()
        self.n_heads = num_heads
        self.dim = dmodel
        self.head_dim = dmodel // num_heads
        self.attention = Attention(dmodel,num_heads,freq_cis,device=device)
        self.feed_forward = FeedForward(
            dim=dmodel,
            hidden_dim= 4 * dmodel,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
            device=device
        )
        
        self.attention_norm = RMSNorm(dmodel, eps=norm_eps, device=device)
        self.ffn_norm = RMSNorm(dmodel, eps=norm_eps,device=device)
        self.freqs_cis = None

    def forward(
        self,
        x: torch.Tensor,
        start_p = 0,
        mask: Optional[torch.Tensor] = None
    ):
        
        h = x + self.attention.forward(
            self.attention_norm(x), start_p, mask
        )
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out



class LLamaEmbedding(nn.Module):
    def __init__(self, vocab_size, dmodel, padding_idx = None, device = "cuda") -> None:
        super().__init__()
        
        self.tok_embeddings = nn.Embedding(vocab_size, dmodel, padding_idx = padding_idx,device=device)
        print("MAKING EMGEDDING WITH ",dmodel)
        print(padding_idx)
        self.vocab_size = vocab_size
    def forward(self, x):
        # print("SHAPE",x.shape)
        return self.tok_embeddings(x)


class LLamaClassification(nn.Module):
    def __init__(self, vocab_size, dmodel, norm_eps=1e-5, type: Literal["cross_entropy", "seq_2_seq"] = "cross_entropy", device = "cuda") -> None:
        super().__init__()
        self.type = type
        self.norm1 = RMSNorm(dmodel, eps=norm_eps,device=device)
        
        self.lm_head = nn.Linear(dmodel, vocab_size, bias=False,device=device)
        self.sfmx = nn.AdaptiveLogSoftmaxWithLoss(dmodel, vocab_size, [100, 1000, 10000],device=device)

    def forward(self, x, targets):
        if self.type == "cross_entropy":
            x = self.norm1(x)
            x = self.lm_head(x)
            x = torch.swapaxes(x, 1, 2)
            return nn.functional.cross_entropy(x, targets)
        elif self.type == "seq_2_seq":
            # from : https://github.com/DS3Lab/DT-FM
            x = self.norm1(x)
            
            shifted_x = x[..., :-1, :].contiguous()
            shifted_targets = targets[..., 1:].contiguous()
            # print(x.shape, shifted_x.shape, shifted_targets.shape, targets.shape)
            return self.sfmx(shifted_x.view(-1, self.sfmx.in_features), shifted_targets.view(-1)).loss
        else:
            raise NotImplemented(f"Not a valid method ${self.type}")
