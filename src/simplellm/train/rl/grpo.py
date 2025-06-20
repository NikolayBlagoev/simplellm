

import torch
from ...llama import LLama
from ...tokenizers.abstracttokenizer import AbstractTokenizer
from dataclasses import dataclass
import torch.nn.functional as F
from typing import List
from ...losses.kl_divergence import kl_loss
from ...losses.grpo_loss import grpo_loss
from torch.utils.data import DataLoader
import random
def _calc_log_probs(net: LLama, x, mask) -> torch.Tensor:
    ret = net(x, mask = mask)
    return F.log_softmax(ret[:, :-1], dim=-1).gather(dim=-1, index=x[:, 1:].unsqueeze(-1)).squeeze(-1)

@dataclass
class Experience:
    outputs: torch.Tensor
    original_log_probs: torch.Tensor
    ref_log_probs: torch.Tensor
    rewards: torch.Tensor
    msk: torch.Tensor
    attention_mask: torch.Tensor
    kl: torch.Tensor


@torch.no_grad()
def rollout(net: LLama, data, completion, loss_func, tokenizer: AbstractTokenizer, repeat = 10, top_k = 5, **kwargs):
    net.eval()
    outputs = net.generate(data.to(net.device), max_new_tokens=512, eos_id=tokenizer.eos_id, pad_id=tokenizer.eos_id, resamples=repeat, top_k = top_k, **kwargs)
    msk = torch.zeros_like(outputs, dtype=torch.bool)
    msk[:, data.shape[1] : ] = True 
    msk[outputs == tokenizer.eos_id] = False
    msk = msk[:, 1:]
    rewards = torch.zeros(repeat, 1, dtype=torch.float)
    loss_func(outputs, msk, completion, tokenizer, rewards)
    return outputs, rewards.to(outputs.device), msk

def GRPO_once(net, original_log_probs, attention_mask):
    
    return _calc_log_probs(net, original_log_probs.to(net.device), attention_mask.to(net.device))

def pre_GRPO(net, net_ref, tokenizer, ds, loss_func, repeat, temperature, **kwargs):
    arr: List[Experience] = []
    for el in ds:
        prompts = el["prompt"]
        completions = el["completion"]
        for p, c in zip(prompts,completions):
            outputs, rewards, msk = rollout(net, p, c, loss_func, tokenizer, repeat, temperature, **kwargs)
            # Normalize rewards:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
            mask = outputs != tokenizer.eos_id
            original_log_probs = _calc_log_probs(net, outputs, mask)
            ref_log_probs = _calc_log_probs(net_ref, outputs, mask)
            kl = kl_loss(original_log_probs, ref_log_probs, msk)
            arr.append(
                Experience(
                    outputs=outputs.to("cpu"),
                    original_log_probs=original_log_probs.to("cpu"),
                    ref_log_probs=ref_log_probs.to("cpu"),
                    rewards=rewards.to("cpu"),
                    msk=msk.to("cpu"),
                    attention_mask=mask.to("cpu"),
                    kl = kl.to("cpu")
                )
            )
            del original_log_probs
            del mask
            del msk
            del rewards
            del outputs
            del ref_log_probs
        
    return arr


def GRPO(net, optimizer, tokenizer, ds, loss_func, repeat, temperature, mb_size = 16, batch_size = 1, epochs = 100, **kwargs):
    arr: List[Experience] = pre_GRPO(net,net,tokenizer,ds,loss_func,repeat,temperature, **kwargs)
    for _ in range(epochs):
        random.shuffle(arr)
        k = 0
        while k < len(arr):
            optimizer.zero_grad()
            if len(arr) - k < mb_size:
                break
            for _ in range(mb_size):
                exp = arr[k]
                k += 1
                log_probs = GRPO_once(net, exp.original_log_probs, exp.attention_mask)
                
                loss = grpo_loss(log_probs, exp.original_log_probs.to(log_probs.device), exp.rewards.to(log_probs.device), exp.msk.to(log_probs.device), kl = kl.to(log_probs.device))
                loss = loss / mb_size
                loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
    return 