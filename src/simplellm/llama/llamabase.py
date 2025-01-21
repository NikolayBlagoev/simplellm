# Adapted form: https://github.com/meta-llama/llama/blob/main/llama/model.py
import math
from typing import Literal, Optional, Tuple
import torch
import torch.nn.functional as F
from torch import nn
from ..utils import *




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
class RoPE(nn.Module):
    def __init__(self, dim, theta=10000, device="cuda"):
        super().__init__()
        # dmodel // num_heads, ctx_size * 2
        
        self.inv_freq = nn.Parameter(1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim)))

    @torch.no_grad()
    def forward(self, x,init_input):
        

        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(init_input.shape[0], -1, 1)
        position_ids = torch.arange(0, init_input.shape[1], device=x.device).unsqueeze(0)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
    
    def rot(self,x):
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


    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

    cos = reshape_for_broadcast(cos, xq_r)
    sin = reshape_for_broadcast(sin, xq_r)
    # print(cos.shape,xq_r.shape)
    xq_out_r = xq_r * cos - xq_i * sin
    xq_out_i = xq_r * sin + xq_i * cos
    xk_out_r = xk_r * cos - xk_i * sin
    xk_out_i = xk_r * sin + xk_i * cos

    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)
    
def repeat_intrleave(x, n):
    if n == 1:
        return x
    b, num_kv, seq_len, head_dim = x.shape
    x = x[:, :, None, :, :].expand(b, num_kv, n, seq_len, head_dim)
    return x.reshape(b, num_kv * n, seq_len, head_dim)

class Attention(nn.Module):
   
    def __init__(self, dmodel, num_heads, ctx_size, num_kv_heads = None, device = "cuda", linear_implementation = "torch"):
        
        super().__init__()
        self.n_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.head_dim = dmodel // num_heads
        self.num_heads = num_heads
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
        
        
        self.rotary_emb = RoPE(dmodel//num_heads,device=device)
        
        

    def forward(
        self,
        x: torch.Tensor,
        start_p: int = 0,
        mask: Optional[torch.Tensor] = None,
        position_embedding = None
    ):
        
        bsz, seqlen, _ = x.shape
        
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        xq = xq.view(bsz, seqlen, self.num_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        cos, sin = position_embedding
        xq, xk = apply_rotary_emb(xq, xk, cos, sin)
        xk = repeat_intrleave(xk, self.num_heads // self.n_kv_heads)
        xv = repeat_intrleave(xv, self.num_heads // self.n_kv_heads)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        scores = torch.nn.functional.scaled_dot_product_attention(
            xq,
            xk,
            xv,
            attn_mask=None,
            dropout_p= 0.0,
            is_causal=True,
        )
        output = scores.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.o_proj(output)


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
            B, T, C = x.shape
            x = x.view(B*T, C)
            targets = targets.view(B*T)
            # x = torch.swapaxes(x, 1, 2)
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
