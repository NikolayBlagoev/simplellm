from simplellm.llama.llamabase import *

import torch
from typing import Union
from torch import nn
import torch.nn.functional as F
from simplellm.utils import IterableModule
class LLamaSeq(IterableModule, nn.ModuleList):
    def forward(self, *inputs):
        x, start_p, mask, position_embeddings = inputs
        for module in self._modules.values():
            x = module(x, start_p, mask, position_embeddings )
        return x
class CausalLLama(IterableModule, nn.Module):
    def __init__(self, vocab_size, dmodel = 4096, num_heads = 32, multiple_of = 256, norm_eps = 1e-6, dropout_prob = 1e2, ctx_size = 2048, num_kv_heads = None, padding_idx = None, device = "cuda", n_layers = 32, ffn_dim_multiplier = None, theta = 10000.0) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, dmodel, padding_idx = padding_idx,device=device)
        
        self.layers = LLamaSeq(
            [
                TransformerBlock(
                    dmodel=dmodel,
                    num_heads=num_heads,
                    ctx_size = ctx_size,
                    multiple_of=multiple_of,
                    norm_eps=norm_eps,
                    ffn_dim_multiplier=ffn_dim_multiplier, 
                    num_kv_heads = num_kv_heads,
                    idx = i,
                    device = device
                ) for i in range(n_layers)
            ])
        self.rotary_emb = RoPE(dmodel // num_heads, theta=theta,device=device)
        self.norm = RMSNorm(dmodel, eps=norm_eps,device=device)
        
    def forward(self, x, start_p = 0, mask = None, position_ids = None, **kwargs):
        B, seq_l = x.shape
        
        h = self.embed_tokens(x)
        position_embeddings = self.rotary_emb(h,B,seq_l)
        h = self.layers(h, start_p, mask, position_embeddings)
        h = self.norm(h)
        return h


class SkipSeq(IterableModule, nn.Sequential):
    def forward(self, *inputs):
        x, start_p, mask, position_embeddings , to_skip = inputs
        for module in self._modules.values():
            if module.idx in to_skip:
                continue
            print("running",module.idx)
            x = module(x, start_p, mask, position_embeddings)
        return x


class SwapSeq(IterableModule, nn.Sequential):
    def forward(self, *inputs):
        x, start_p, mask, position_embeddings, order = inputs
        #print(order)
        for v in order:
            #print(v)
            module = self._modules[str(v)]
            
            x = module(x, start_p, mask, position_embeddings)
        return x

class SkipLLama(IterableModule, nn.Module):
    def __init__(self, vocab_size, dmodel = 4096, num_heads = 32, multiple_of = 256, norm_eps = 1e-6, dropout_prob = 1e2, ctx_size = 2048, num_kv_heads = None, padding_idx = None, device = "cuda", n_layers = 32, ffn_dim_multiplier = None, theta = 10000.0) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, dmodel, padding_idx = padding_idx,device=device)
        
        self.layers = SkipSeq(
            *[
                TransformerBlock(
                    dmodel=dmodel,
                    num_heads=num_heads,
                    ctx_size = ctx_size,
                    multiple_of=multiple_of,
                    norm_eps=norm_eps,
                    ffn_dim_multiplier=ffn_dim_multiplier, 
                    num_kv_heads = num_kv_heads,
                    idx = i,
                    device = device
                ) for i in range(n_layers)
            ])
        self.rotary_emb = RoPE(dmodel // num_heads, theta=theta,device=device)
        self.norm = RMSNorm(dmodel, eps=norm_eps,device=device)
        
    def forward(self, x,  start_p = 0, mask = None, position_ids = None,to_skip = [], **kwargs):
        B, seq_l = x.shape
        
        h = self.embed_tokens(x)
        position_embeddings = self.rotary_emb(h,B,seq_l)
        h = self.layers(h,start_p,mask,position_embeddings,to_skip)
        h = self.norm(h)
        return h


class SwapLLama(IterableModule, nn.Module):
    def __init__(self, vocab_size, dmodel = 4096, num_heads = 32, multiple_of = 256, norm_eps = 1e-6, dropout_prob = 1e2, ctx_size = 2048, num_kv_heads = None, padding_idx = None, device = "cuda", n_layers = 32, ffn_dim_multiplier = None, theta = 10000.0) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, dmodel, padding_idx = padding_idx,device=device)
        
        self.layers = SwapSeq(
            *[
                TransformerBlock(
                    dmodel=dmodel,
                    num_heads=num_heads,
                    ctx_size = ctx_size,
                    multiple_of=multiple_of,
                    norm_eps=norm_eps,
                    ffn_dim_multiplier=ffn_dim_multiplier, 
                    num_kv_heads=num_kv_heads,
                    idx = i,
                    device = device
                ) for i in range(n_layers)
            ]
        )
        
            
        self.rotary_emb = RoPE(dmodel // num_heads, theta=theta,device=device)
        self.norm = RMSNorm(dmodel, eps=norm_eps,device=device)
        
    def forward(self, x, start_p = 0, mask = None, position_ids = None, order = [], **kwargs):
        B, seq_l = x.shape
        h = self.embed_tokens(x)
        position_embeddings = self.rotary_emb(h,B,seq_l)
        
        h = self.layers(h, start_p, mask, position_embeddings, order)
        h = self.norm(h)
       
        return h

    
class LLama(IterableModule, nn.Module):
    def __init__(self, mdl_type: Union[SwapLLama,SkipLLama,CausalLLama],vocab_size, dmodel = 4096, num_heads = 32, multiple_of = 256, norm_eps = 1e-6, dropout_prob = 1e2, ctx_size = 2048, num_kv_heads = None, padding_idx = None, device = "cuda", n_layers = 32, ffn_dim_multiplier = None, shared = False):
        super().__init__()
        self.max_seq = ctx_size
        self.device = device
        self.model = mdl_type(vocab_size,dmodel,num_heads,multiple_of,norm_eps,dropout_prob,ctx_size,num_kv_heads,padding_idx,device,n_layers,ffn_dim_multiplier)
        self.lm_head = nn.Linear(dmodel, vocab_size, bias=False,device=device)
        # self.lm_head = nn.AdaptiveLogSoftmaxWithLoss(dmodel, vocab_size, [1000, 2000, 5000],device=device)
        if shared:
            self.model.embed_tokens.weight = self.lm_head.weight
        self._initialize()
    def forward(self, x, **kwargs):
        #print(*args) 
        return self.lm_head(self.model(x,**kwargs))

    @torch.inference_mode()
    def generate(self, inp, tokenizer, max_gen_len: int, *args):
        pad_id = tokenizer.pad_id
        tokens = torch.full((1, self.max_seq), pad_id, dtype=torch.long, device=self.device)
        tokens[1,: inp.shape[0]] = inp
        head = 0
        eos_reached = False
        input_text_mask = tokens != pad_id
        

        for cur_pos in range(inp.shape[0], inp.shape[0] + max_gen_len):
            logits = self.model.forward(tokens[:, head:cur_pos],*args)
            next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)
            
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            
            eos_reached |= (~input_text_mask[:, cur_pos]) & (
                next_token == self.tokenizer.eos_id
            )
            
            if all(eos_reached):
                break

        
        out_tokens = []
        for i, toks in enumerate(tokens.tolist()):

            start = 0
            toks = toks[start : inp.shape[0] + max_gen_len]
            
            
            if self.tokenizer.eos_id in toks:
                eos_idx = toks.index(self.tokenizer.eos_id)
                toks = toks[:eos_idx]
                
            out_tokens.append(toks)
            
        return out_tokens
    

class LLamaStage(IterableModule, nn.Module):
    def __init__(self, dmodel = 4096, num_heads = 32, multiple_of = 256, norm_eps = 1e-6, dropout_prob = 1e2, ctx_size = 2048, n_layers = 4, num_kv_heads = None, padding_idx = None, device = "cuda", ffn_dim_multiplier = None, linear_implementation = "torch", theta = 10000.0) -> None:
        super().__init__()
        
        
        self.layers = SkipSeq(
            *[
                TransformerBlock(
                    dmodel=dmodel,
                    num_heads=num_heads,
                    ctx_size = ctx_size,
                    multiple_of=multiple_of,
                    norm_eps=norm_eps,
                    ffn_dim_multiplier=ffn_dim_multiplier, 
                    num_kv_heads = num_kv_heads,
                    idx = i,
                    device = device,
                    linear_implementation = linear_implementation
                ) for i in range(n_layers)
            ]
        
        )
        self.rotary_emb = RoPE(dmodel // num_heads, theta=theta,device=device)
        self._initialize()
        
    
    def forward(self, x, start_p = 0):
        B, seq_l, _ = x.shape
        position_embeddings = self.rotary_emb(x,B,seq_l)
        
        h = self.layers(x,start_p,None,position_embeddings,[])
        
        return h

class LLamaFirstStage(IterableModule, nn.Module):
    def __init__(self, vocab_size, dmodel, num_heads, n_layers = 4, multiple_of = 256, norm_eps = 1e-6, ffn_dim_multiplier = None, ctx_size = 2048, num_kv_heads = None, padding_idx = None, device = "cuda", linear_implementation = "torch", share_weights = False, theta = 10000.0, de_embed = True) -> None:
        super().__init__()
        self._embedding = LLamaEmbedding(vocab_size,dmodel,padding_idx=padding_idx,device=device)
        
        if de_embed:
            self.norm = RMSNorm(dmodel, eps=norm_eps,device=device)
            self.lm_head = nn.Linear(dmodel, vocab_size, bias=False,device=device)
            if share_weights:
                self._embedding.weight = self.lm_head.weight
        self.n_layers = n_layers
        if self.n_layers > 0:
            self.layers = SkipSeq(
                *[
                    TransformerBlock(
                        dmodel=dmodel,
                        num_heads=num_heads,
                        ctx_size = ctx_size,
                        multiple_of=multiple_of,
                        norm_eps=norm_eps,
                        ffn_dim_multiplier=ffn_dim_multiplier, 
                        num_kv_heads = num_kv_heads,
                        idx = i,
                        device = device,
                        linear_implementation = linear_implementation
                    ) for i in range(n_layers)
                ]
            
            )
            self.rotary_emb = RoPE(dmodel // num_heads, theta=theta,device=device)
        self._initialize()
        
        
    def embed(self, x):
        x = self._embedding(x)
        if self.n_layers == 0:
            return x
        B, seq_l, _ = x.shape
        position_embeddings = self.rotary_emb(x,B,seq_l)
        
        h = self.layers(x,0,None,position_embeddings,[])
        
        return h
    def forward_end(self, x):
       
        return  self.lm_head(self.norm(x))

class LLamaLastStage(IterableModule, nn.Module):
    def __init__(self, vocab_size, dmodel = 4096, num_heads = 32, multiple_of = 256, norm_eps = 1e-6, dropout_prob = 1e2, ctx_size = 2048, n_layers = 4, num_kv_heads = None, padding_idx = None, device = "cuda", ffn_dim_multiplier = None, linear_implementation = "torch", theta = 10000.0) -> None:
        super().__init__()
        
        
        self.layers = SkipSeq(
            *[
                TransformerBlock(
                    dmodel=dmodel,
                    num_heads=num_heads,
                    ctx_size = ctx_size,
                    multiple_of=multiple_of,
                    norm_eps=norm_eps,
                    ffn_dim_multiplier=ffn_dim_multiplier, 
                    num_kv_heads = num_kv_heads,
                    idx = i,
                    device = device,
                    linear_implementation = linear_implementation
                ) for i in range(n_layers)
            ]
        
        )
        self.norm = RMSNorm(dmodel, eps=norm_eps,device=device)
        self.lm_head = nn.Linear(dmodel, vocab_size, bias=False,device=device)
        self.rotary_emb = RoPE(dmodel // num_heads, theta=theta,device=device)
        self._initialize()
        
    def forward(self, x, start_p = 0):
        B, seq_l, _ = x.shape
        position_embeddings = self.rotary_emb(x,B,seq_l)
        
        h = self.layers(x,start_p,None,position_embeddings,[])
        
        return self.lm_head(self.norm(h))