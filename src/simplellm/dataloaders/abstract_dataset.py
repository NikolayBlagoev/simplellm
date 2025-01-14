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
            
            while len(tmp) >= self.seq_length + 2:
                tmp_x = tmp[:self.seq_length]
                tmp_trgt = tmp[1:self.seq_length+1]
                tmp =  tmp[self.seq_length+1:]
               
                yield torch.tensor(tmp_x),torch.tensor(tmp_trgt)

    def get_stream(self):
        return cycle(self.get_data())

    def __iter__(self):
        return self.get_stream()