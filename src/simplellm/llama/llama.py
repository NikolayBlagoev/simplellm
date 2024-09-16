from simplellm.llama.llamabase import *

import torch
from typing import Union
from torch import nn
import torch.nn.functional as F
class CausalLLama(nn.Module):
    def __init__(self, vocab_size, dmodel = 4096, num_heads = 32, multiple_of = 256, norm_eps = 1e-5, dropout_prob = 1e2, ctx_size = 2048, padding_idx = None, device = "cuda", n_layers = 32, ffn_dim_multiplier = None) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, dmodel, padding_idx = padding_idx,device=device)
        
        self.layers = nn.Sequential(
            *[
                TransformerBlock(
                    dmodel=dmodel,
                    num_heads=num_heads,
                    multiple_of=multiple_of,
                    norm_eps=norm_eps,
                    ffn_dim_multiplier=ffn_dim_multiplier, 
                    idx = i,
                    device = device
                ) for i in range(n_layers)
            ])
        self.norm = RMSNorm(dmodel, eps=norm_eps,device=device)
        
    def forward(self, x, *args):
        _, seq_l = x.shape
        h = self.embed_tokens(x)
        
        h = self.layers(h)
        h = self.norm(h)
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
        x, start_p, mask, order = inputs

        for v in order:
            if self._modules.get(v) == None:
                continue
            module = self._modules[v]
            
            x = module(x, start_p, mask)
        return x

class SkipLLama(nn.Module):
    def __init__(self, vocab_size, dmodel = 4096, num_heads = 32, multiple_of = 256, norm_eps = 1e-5, dropout_prob = 1e2, ctx_size = 2048, padding_idx = None, device = "cuda", n_layers = 32, ffn_dim_multiplier = None) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, dmodel, padding_idx = padding_idx,device=device)
        
        self.layers = SkipSeq(
            *[
                TransformerBlock(
                    dmodel=dmodel,
                    num_heads=num_heads,
                    
                    multiple_of=multiple_of,
                    norm_eps=norm_eps,
                    ffn_dim_multiplier=ffn_dim_multiplier, 
                    idx = i,
                    device = device
                ) for i in range(n_layers)
            ])
        self.norm = RMSNorm(dmodel, eps=norm_eps,device=device)
        
    def forward(self, x, to_skip = []):
        _, seq_l = x.shape
        h = self.embed_tokens(x)
        
        h = self.layers(h,0,None,to_skip)
        h = self.norm(h)
        return h


class SwapLLama(nn.Module):
    def __init__(self, vocab_size, dmodel = 4096, num_heads = 32, multiple_of = 256, norm_eps = 1e-5, dropout_prob = 1e2, ctx_size = 2048, padding_idx = None, device = "cuda", n_layers = 32, ffn_dim_multiplier = None) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, dmodel, padding_idx = padding_idx,device=device)
        
        self.layers = SwapSeq(
            *[
                TransformerBlock(
                    dmodel=dmodel,
                    num_heads=num_heads,
                    
                    multiple_of=multiple_of,
                    norm_eps=norm_eps,
                    ffn_dim_multiplier=ffn_dim_multiplier, 
                    idx = i,
                    device = device
                ) for i in range(n_layers)
            ])
        self.norm = RMSNorm(dmodel, eps=norm_eps,device=device)
        
    def forward(self, x, order = []):
        _, seq_l = x.shape
        h = self.embed_tokens(x)
        
        h = self.layers(h,0,None,order)
        h = self.norm(h)
       
        return h

    
class LLama(nn.Module):
    def __init__(self, mdl_type: Union[SwapLLama,SkipLLama,CausalLLama],vocab_size, dmodel = 4096, num_heads = 32, multiple_of = 256, norm_eps = 1e-5, dropout_prob = 1e2, ctx_size = 2048, padding_idx = None, device = "cuda", n_layers = 32, ffn_dim_multiplier = None):
        super().__init__()
        self.max_seq = ctx_size
        self.device = device
        self.model = mdl_type(vocab_size,dmodel,num_heads,multiple_of,norm_eps,dropout_prob,ctx_size,padding_idx,device,n_layers,ffn_dim_multiplier)
        self.lm_head = nn.Linear(dmodel, vocab_size, bias=False,device=device)

    def forward(self, x, *args):
        return self.lm_head(self.model(x,args))

    @torch.inference_mode()
    def generate(self, inp, tokenizer, max_gen_len: int, *args):
        pad_id = tokenizer.pad_id
        tokens = torch.full((1, self.max_seq), pad_id, dtype=torch.long, device=self.device)
        tokens[: inp.shape[0]] = inp
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