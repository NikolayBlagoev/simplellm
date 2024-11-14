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
    
def repeat_intrleave(x, n):
    if n == 1:
        return x
    b, num_kv, seq_len, head_dim = x.shape
    x = x[:, :, None, :, :].expand(b, num_kv, n, seq_len, head_dim)
    return x.reshape(b, num_kv * n, seq_len, head_dim)

class Attention(nn.Module):
   
    def __init__(self, dmodel, num_heads, ctx_size, num_kv_heads = None, device = "cuda"):
        
        super().__init__()
        self.n_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.head_dim = dmodel // num_heads
        self.num_heads = num_heads
        self.q_proj = nn.Linear(
            dmodel,
            num_heads * self.head_dim,
            bias=False,
            device=device
        )
        self.k_proj = nn.Linear(
            dmodel,
            self.n_kv_heads * self.head_dim,
            bias=False,
            device=device
        )
        self.v_proj = nn.Linear(
            dmodel,
            self.n_kv_heads * self.head_dim,
            bias=False,
            device=device
        )
        self.o_proj = nn.Linear(
            num_heads * self.head_dim,
            dmodel,
            bias=False,
            device=device
        )
        
        
        self.rotary_emb = RoPE(dmodel//num_heads,device=device)
        mask = torch.full((1, 1, ctx_size, ctx_size), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer("mask", mask)
        

    def forward(
        self,
        x: torch.Tensor,
        start_p: int = 0,
        mask: Optional[torch.Tensor] = None,
        position_embedding = None
    ):
        
        bsz, seqlen, _ = x.shape
        
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        xq = xq.view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim).transpose(1, 2)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim).transpose(1, 2)
        if position_embedding is None:
            cos, sin = self.rotary_emb(xv,x)
        else:
            cos, sin = position_embedding
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        xq = (xq * cos) + (self.rotary_emb.rot(xq) * sin)
        xk = (xk * cos) + (self.rotary_emb.rot(xk) * sin)
        xk = repeat_intrleave(xk, self.num_heads // self.n_kv_heads)
        xv = repeat_intrleave(xv, self.num_heads // self.n_kv_heads)
        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        else:
            scores = scores + self.mask[:, :, :seqlen, :seqlen].to(x.device)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, xv)  # (bs, n_local_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.o_proj(output)


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

        self.gate_proj = nn.Linear(
            dim, hidden_dim, bias=False,device=device)
        self.down_proj = nn.Linear(
            hidden_dim, dim, bias=False,device=device)
        self.up_proj = nn.Linear(
            dim, hidden_dim, bias=False,device=device)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, dmodel, num_heads, ctx_size, multiple_of = 256, norm_eps = 1e-5, ffn_dim_multiplier = None, idx = None, device = "cuda"):
        super().__init__()
        self.n_heads = num_heads
        self.dim = dmodel
        self.head_dim = dmodel // num_heads
        self.self_attn = Attention(dmodel,num_heads,ctx_size,device=device)
        self.mlp = FeedForward(
            dim=dmodel,
            hidden_dim= 4 * dmodel,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
            device=device
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
        mask: Optional[torch.Tensor] = None
    ):
        
        h = x + self.self_attn.forward(
            self.input_layernorm(x), start_p, mask
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
