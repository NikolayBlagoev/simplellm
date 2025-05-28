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
    def __init__(self, vocab_size, dmodel = 4096, num_heads = 32, hidden_dim = None, norm_eps = 1e-6, dropout_prob = 0, head_dim = None, ctx_size = 2048, num_kv_heads = None, padding_idx = None, device = "cuda", n_layers = 32, theta = 10000.0, rope_parameters = None) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, dmodel, padding_idx = padding_idx,device=device)
        
        self.layers = LLamaSeq(
            [
                TransformerBlock(
                    dmodel=dmodel,
                    num_heads=num_heads,
                    ctx_size = ctx_size,
                    hidden_dim=hidden_dim,
                    norm_eps=norm_eps,
                    head_dim = head_dim,
                    num_kv_heads = num_kv_heads,
                    idx = i,
                    dropout_prob = dropout_prob,
                    device = device
                ) for i in range(n_layers)
            ])
        if rope_parameters == None:
            self.rotary_emb = RoPE(initializing_function="linear",dim=dmodel // num_heads, theta=theta,device=device)
        else:
            self.rotary_emb = RoPE(**rope_parameters)
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
            # print("running",module.idx)
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
    def __init__(self, vocab_size, dmodel = 4096, num_heads = 32, hidden_dim = None, norm_eps = 1e-6, dropout_prob = 0, head_dim = None, ctx_size = 2048, num_kv_heads = None, padding_idx = None, device = "cuda", n_layers = 32, theta = 10000.0, rope_parameters = None) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, dmodel, padding_idx = padding_idx,device=device)
        
        self.layers = SkipSeq(
            *[
                TransformerBlock(
                    dmodel=dmodel,
                    num_heads=num_heads,
                    ctx_size = ctx_size,
                    hidden_dim=hidden_dim,
                    norm_eps=norm_eps,
                    num_kv_heads = num_kv_heads,
                    idx = i,
                    dropout_prob = dropout_prob,
                    head_dim = head_dim,
                    device = device
                ) for i in range(n_layers)
            ])
        if rope_parameters == None:
            self.rotary_emb = RoPE(initializing_function="linear",dim=dmodel // num_heads, theta=theta,device=device)
        else:
            self.rotary_emb = RoPE(**rope_parameters)
        self.norm = RMSNorm(dmodel, eps=norm_eps,device=device)
        
    def forward(self, x,  start_p = 0, mask = None, position_ids = None,to_skip = [], **kwargs):
        B, seq_l = x.shape
        
        h = self.embed_tokens(x)
        position_embeddings = self.rotary_emb(h,B,seq_l)
        h = self.layers(h,start_p,mask,position_embeddings,to_skip)
        h = self.norm(h)
        return h


class SwapLLama(IterableModule, nn.Module):
    def __init__(self, vocab_size, dmodel = 4096, num_heads = 32, hidden_dim = None, norm_eps = 1e-6, dropout_prob = 0, head_dim = None, ctx_size = 2048, num_kv_heads = None, padding_idx = None, device = "cuda", n_layers = 32, theta = 10000.0, rope_parameters = None) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, dmodel, padding_idx = padding_idx,device=device)
        
        self.layers = SwapSeq(
            *[
                TransformerBlock(
                    dmodel=dmodel,
                    num_heads=num_heads,
                    ctx_size = ctx_size,
                    hidden_dim=hidden_dim,
                    norm_eps=norm_eps, 
                    num_kv_heads=num_kv_heads,
                    idx = i,
                    dropout_prob = dropout_prob,
                    head_dim = head_dim,
                    device = device
                ) for i in range(n_layers)
            ]
        )
        
            
        if rope_parameters == None:
            self.rotary_emb = RoPE(initializing_function="linear",dim=dmodel // num_heads, theta=theta,device=device)
        else:
            self.rotary_emb = RoPE(**rope_parameters)
        self.norm = RMSNorm(dmodel, eps=norm_eps,device=device)
        
    def forward(self, x, start_p = 0, mask = None, position_ids = None, order = [], **kwargs):
        B, seq_l = x.shape
        h = self.embed_tokens(x)
        position_embeddings = self.rotary_emb(h,B,seq_l)
        
        h = self.layers(h, start_p, mask, position_embeddings, order)
        h = self.norm(h)
       
        return h

    
class LLama(IterableModule, nn.Module):
    def __init__(self, mdl_type: Union[SwapLLama,SkipLLama,CausalLLama],vocab_size, dmodel = 4096, num_heads = 32, hidden_dim = None, norm_eps = 1e-6, dropout_prob = 0, head_dim = None, ctx_size = 2048, num_kv_heads = None, padding_idx = None, device = "cuda", n_layers = 32, shared = False, rope_parameters = None):
        super().__init__()
        self.max_seq = ctx_size
        self.device = device
        self.model = mdl_type(vocab_size=vocab_size,dmodel=dmodel,num_heads=num_heads,hidden_dim=hidden_dim,norm_eps=norm_eps, head_dim = head_dim, dropout_prob=dropout_prob,ctx_size=ctx_size,num_kv_heads=num_kv_heads,padding_idx=padding_idx,device=device,n_layers=n_layers,rope_parameters = rope_parameters)
        self.lm_head = nn.Linear(dmodel, vocab_size, bias=False,device=device)
        # self.lm_head = nn.AdaptiveLogSoftmaxWithLoss(dmodel, vocab_size, [1000, 2000, 5000],device=device)
        if shared:
            self.model.embed_tokens.weight = self.lm_head.weight
        self._initialize()
    def forward(self, x, **kwargs):
        #print(*args) 
        return self.lm_head(self.model(x,**kwargs))

    @torch.no_grad()
    def _expand(self, inp: torch.Tensor, resample):
        if resample == 1:
            return inp
        inp = inp.repeat_interleave(resample,dim = 0)
        return inp
    @torch.inference_mode()
    def generate(self, inp, max_new_tokens, temperature=1.0, top_k=None, resamples = 1, pad_id = 0, eos_id = None, **kwargs):
        inp = self._expand(inp,resample)
        if eos_id != None:
            unfinished_sequences = torch.ones(inp.shape[0], dtype=torch.long, device=inp.device)
        
        start_pos = inp.shape[1]
        for cur_pos in range(max_new_tokens):
            outputs = self.forward(inp,**kwargs)
            next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=inp.device)
            if temperature == 0:
                _, next_token = torch.topk(next_token_logits, k=1, dim=-1)
            else:
                next_token_logits /= temperature
                if top_k is not None:
                    v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            if eos_id != None:
                next_token = next_token * unfinished_sequences + pad_id * (1 - unfinished_sequences)
            inp = torch.cat([inp, next_token[:, None]], dim=-1)
            if eos_id != None:
                unfinished_sequences = unfinished_sequences & (next_token == eos_id)
                if unfinished_sequences.sum() == 0:
                    break 
            del outputs
        return inp

    

class LLamaStage(IterableModule, nn.Module):
    def __init__(self, dmodel = 4096, num_heads = 32, hidden_dim = None, norm_eps = 1e-6, dropout_prob = 0, ctx_size = 2048, n_layers = 4, head_dim = None, num_kv_heads = None, padding_idx = None, device = "cuda", linear_implementation = "torch", theta = 10000.0, rope_parameters = None) -> None:
        super().__init__()
        
        
        self.layers = SkipSeq(
            *[
                TransformerBlock(
                    dmodel=dmodel,
                    num_heads=num_heads,
                    ctx_size = ctx_size,
                    hidden_dim=hidden_dim,
                    norm_eps=norm_eps,
                    num_kv_heads = num_kv_heads,
                    idx = i,
                    dropout_prob = dropout_prob,
                    head_dim = head_dim,
                    device = device,
                    linear_implementation = linear_implementation
                ) for i in range(n_layers)
            ]
        
        )
        if rope_parameters == None:
            self.rotary_emb = RoPE(initializing_function="linear",dim=dmodel // num_heads, theta=theta,device=device)
        else:
            self.rotary_emb = RoPE(**rope_parameters)
        self._initialize()
        
    
    def forward(self, x, start_p = 0):
        B, seq_l, _ = x.shape
        position_embeddings = self.rotary_emb(x,B,seq_l)
        
        h = self.layers(x,start_p,None,position_embeddings,[])
        
        return h

class LLamaFirstStage(IterableModule, nn.Module):
    def __init__(self, vocab_size, dmodel, num_heads, n_layers = 4, hidden_dim = None, norm_eps = 1e-6, dropout_prob = 0, ctx_size = 2048, num_kv_heads = None, head_dim = None, padding_idx = None, device = "cuda", linear_implementation = "torch", share_weights = False, theta = 10000.0, de_embed = True, rope_parameters = None) -> None:
        super().__init__()
        self._embedding = nn.Embedding(vocab_size, dmodel, padding_idx = padding_idx,device=device)
        
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
                        hidden_dim=hidden_dim,
                        norm_eps=norm_eps,
                        num_kv_heads = num_kv_heads,
                        idx = i,
                        dropout_prob = dropout_prob,
                        head_dim = head_dim,
                        device = device,
                        linear_implementation = linear_implementation
                    ) for i in range(n_layers)
                ]
            
            )
            if rope_parameters == None:
                self.rotary_emb = RoPE(initializing_function="linear",dim=dmodel // num_heads, theta=theta,device=device)
            else:
                self.rotary_emb = RoPE(**rope_parameters)
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
    def __init__(self, vocab_size, dmodel = 4096, num_heads = 32, hidden_dim = None, norm_eps = 1e-6, dropout_prob = 0, head_dim = None, ctx_size = 2048, n_layers = 4, num_kv_heads = None, padding_idx = None, device = "cuda", linear_implementation = "torch", theta = 10000.0, rope_parameters = None) -> None:
        super().__init__()
        
        
        self.layers = SkipSeq(
            *[
                TransformerBlock(
                    dmodel=dmodel,
                    num_heads=num_heads,
                    ctx_size = ctx_size,
                    hidden_dim=hidden_dim,
                    norm_eps=norm_eps,
                    num_kv_heads = num_kv_heads,
                    idx = i,
                    dropout_prob = dropout_prob,
                    head_dim = head_dim,
                    device = device,
                    linear_implementation = linear_implementation
                ) for i in range(n_layers)
            ]
        
        )
        self.norm = RMSNorm(dmodel, eps=norm_eps,device=device)
        self.lm_head = nn.Linear(dmodel, vocab_size, bias=False,device=device)
        if rope_parameters == None:
            self.rotary_emb = RoPE(initializing_function="linear",dim=dmodel // num_heads, theta=theta,device=device)
        else:
            self.rotary_emb = RoPE(**rope_parameters)
        self._initialize()
        
    def forward(self, x, start_p = 0):
        B, seq_l, _ = x.shape
        position_embeddings = self.rotary_emb(x,B,seq_l)
        
        h = self.layers(x,start_p,None,position_embeddings,[])
        
        return self.lm_head(self.norm(h))