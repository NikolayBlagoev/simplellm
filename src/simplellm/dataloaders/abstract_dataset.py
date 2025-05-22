from itertools import cycle
import torch
from datasets import load_dataset
from simplellm.tokenizers.abstracttokenizer import AbstractTokenizer
from torch.utils.data import DataLoader, IterableDataset

class AbstractDataset(IterableDataset):
    def __init__(self, dataset, tokenizer, seq_length=2048):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.seq_length = seq_length

    def get_data(self):
        tmp = []
        for txt in self.dataset:
            
            tmp += [self.tokenizer.bos_id] + txt['text']
            
            while len(tmp) >= self.seq_length:
                tmp_x = tmp[:self.seq_length]
                tmp =  tmp[self.seq_length:]
                to_yield = torch.tensor(tmp_x)
                yield to_yield
            tmp = []
            

    def get_stream(self):
        return cycle(self.get_data())

    def __iter__(self):
        return self.get_stream()