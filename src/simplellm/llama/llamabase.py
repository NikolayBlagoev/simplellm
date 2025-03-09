# Adapted form: https://github.com/meta-llama/llama/blob/main/llama/model.py
import math
from typing import Literal, Optional, Tuple
import torch
import torch.nn.functional as F
from torch import nn
from ..utils import *

# CORRECT
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
# CORRECT
class RoPE(nn.Module):
    def __init__(self, dim, theta=10000.0, device="cuda"):
        super().__init__()
        
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq
        
    @torch.no_grad()
    def forward(self, x,B,SQLEN):
        

        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(B, -1, 1)
        position_ids = torch.arange(0, SQLEN, device=x.device).unsqueeze(0)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
    
    def rot(x):
        f = x[..., : x.shape[-1] // 2]
        s = x[..., x.shape[-1] // 2 :]
        return torch.cat((-s, f), dim=-1)

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cos = torch.cos(freqs)  # real part
    freqs_sin = torch.sin(freqs)  # imaginary part
    return freqs_cos, freqs_sin

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    # print(x.shape,freqs_cis.shape)
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:

    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    # print(cos.shape,xq_r.shape)
    xq = (xq * cos) + (RoPE.rot(xq) * sin)
    xk = (xk * cos) + (RoPE.rot(xk) * sin)
    return xq, xk
    
def repeat_intrleave(x, n):
    if n == 1:
        return x
    b, num_kv, seq_len, head_dim = x.shape
    x = x[:, :, None, :, :].expand(b, num_kv, n, seq_len, head_dim)
    return x.reshape(b, num_kv * n, seq_len, head_dim)

class Attention(nn.Module):
   
    def __init__(self, dmodel, num_heads, ctx_size, num_kv_heads = None, device = "cuda", linear_implementation = "torch", drop_out_p = 0.0):
        
        super().__init__()
        self.n_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.head_dim = dmodel // num_heads
        self.num_heads = num_heads
        self.drop_out_p = drop_out_p
        self.scaling = self.head_dim**-0.5
        if linear_implementation == "torch":
            linear_implementation = nn.Linear
        elif linear_implementation == "delayed":
            linear_implementation = CustomLinear
        else:
            raise AttributeError(f"UNKNOWN IMPLEMENTATION {linear_implementation}")
        self.q_proj = linear_implementation(
            dmodel,
            num_heads * self.head_dim,
            bias=False,
            device=device
        )
        self.k_proj = linear_implementation(
            dmodel,
            self.n_kv_heads * self.head_dim,
            bias=False,
            device=device
        )
        self.v_proj = linear_implementation(
            dmodel,
            self.n_kv_heads * self.head_dim,
            bias=False,
            device=device
        )
        self.o_proj = linear_implementation(
            num_heads * self.head_dim,
            dmodel,
            bias=False,
            device=device
        )
        
        # TODO: CHECK
        # self.rotary_emb = RoPE(dmodel//num_heads,device=device)
        
        

    def forward(
        self,
        x: torch.Tensor,
        start_p: int = 0,
        mask: Optional[torch.Tensor] = None,
        position_embedding = None
    ):
        
        bsz, seqlen, _ = x.shape
        
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        xq = xq.view(bsz, seqlen, -1, self.head_dim)
        xk = xk.view(bsz, seqlen, -1, self.head_dim)
        xv = xv.view(bsz, seqlen, -1, self.head_dim)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        if position_embedding:
            cos, sin = position_embedding
        # TODO: ADD ELSE


        xq, xk = apply_rotary_emb(xq, xk, cos, sin)
        
        xk = repeat_intrleave(xk, self.num_heads // self.n_kv_heads)
        xv = repeat_intrleave(xv, self.num_heads // self.n_kv_heads)
        xq = xq.contiguous()
        xv = xv.contiguous()
        xk = xk.contiguous()
        # TODO: Implement self...
        
        o = F.scaled_dot_product_attention(
            xq,
            xk,
            xv,
            attn_mask=None,
            is_causal=True
        ).transpose(1, 2).contiguous()
        o = o.reshape(bsz, seqlen, -1).contiguous()
        o = self.o_proj(o)
        
        return o


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
        device = "cuda",
        linear_implementation = "torch"
    ):
        
        super().__init__()
        hidden_dim = 4 * dim
        
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
       
        if linear_implementation == "torch":
            linear_implementation = nn.Linear
        elif linear_implementation == "delayed":
            linear_implementation = CustomLinear
        else:
            raise AttributeError(f"UNKNOWN IMPLEMENTATION {linear_implementation}")
        self.gate_proj = linear_implementation(
            dim, hidden_dim, bias=False,device=device)
        self.down_proj =  linear_implementation(
            hidden_dim, dim, bias=False,device=device)
        self.up_proj = linear_implementation(
            dim, hidden_dim, bias=False,device=device)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

# CHECKED
class TransformerBlock(nn.Module):
    def __init__(self, dmodel, num_heads, ctx_size, multiple_of = 256, norm_eps = 1e-5, ffn_dim_multiplier = None, num_kv_heads = None, idx = None, device = "cuda", linear_implementation = "torch"):
        super().__init__()
        self.n_heads = num_heads
        self.dim = dmodel
        self.head_dim = dmodel // num_heads
        self.self_attn = Attention(dmodel,num_heads,ctx_size,device=device,linear_implementation=linear_implementation,num_kv_heads=num_kv_heads)
        self.mlp = FeedForward(
            dim=dmodel,
            hidden_dim= dmodel,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
            device=device,
            linear_implementation=linear_implementation
        )
        if idx == None:
            raise ValueError("Index cannot be none!")
        self.idx = idx
        self.input_layernorm = RMSNorm(dmodel, eps=norm_eps, device=device)
        self.post_attention_layernorm = RMSNorm(dmodel, eps=norm_eps,device=device)
        self.freqs_cis = None

    def forward(
        self,
        x: torch.Tensor,
        start_p = 0,
        mask: Optional[torch.Tensor] = None,
        position_embedding = None
    ):
        #print(self.idx)
        h = x + self.self_attn.forward(
            self.input_layernorm(x), start_p, mask, position_embedding=position_embedding
        )
        out = h + self.mlp.forward(self.post_attention_layernorm(h))
        return out



class LLamaEmbedding(nn.Module):
    def __init__(self, vocab_size, dmodel, padding_idx = None, device = "cuda") -> None:
        super().__init__()
        
        self.tok_embeddings = nn.Embedding(vocab_size, dmodel, padding_idx = padding_idx,device=device)
        self.vocab_size = vocab_size
    def forward(self, x):
        # print("SHAPE",x.shape)
        return self.tok_embeddings(x)

