from itertools import cycle
import torch
from datasets import load_dataset
from simplellm.tokenizers.abstracttokenizer import AbstractTokenizer
from torch.utils.data import DataLoader, IterableDataset

class FinetuneDataset(IterableDataset):
    def __init__(self, dataset, tokenizer: AbstractTokenizer, seq_length=2048):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.seq_length = seq_length

    def get_data(self):
        
        for txt in self.dataset:
            
            tmp = [self.tokenizer.bos_id] + txt['text']
            
            while len(tmp) >= self.seq_length:
                tmp_x = tmp[:self.seq_length-1] + [self.tokenizer.eos_id]
                tmp =  tmp[self.seq_length-1:]
                to_yield = torch.tensor(tmp_x)
                yield to_yield, torch.ones_like(to_yield)
            
            lg = len(tmp)
            tmp += [self.tokenizer.pad_id for _ in range(self.seq_length - lg)]
            to_yield = torch.tensor(tmp)
            mask = torch.ones_like(to_yield)
            mask[lg:] = 0
            yield to_yield, mask
            
            

    def get_stream(self):
        return cycle(self.get_data())

    def __iter__(self):
        return self.get_stream()