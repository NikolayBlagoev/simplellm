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
        
        for txt in self.dataset:
            ret = [self.tokenizer.bos_id]
            trgt = []
            ret += txt['text']
            trgt += txt['text']
            while len(ret) >= self.seq_length + 1:
                tmp_x = ret[:self.seq_length]
                tmp_trgt = trgt[:self.seq_length]
                ret =  ret[self.seq_length:]
                trgt = trgt[self.seq_length:]
                yield torch.tensor(tmp_x),torch.tensor(tmp_trgt)

    def get_stream(self):
        return cycle(self.get_data())

    def __iter__(self):
        return self.get_stream()