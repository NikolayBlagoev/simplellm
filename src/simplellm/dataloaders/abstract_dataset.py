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
        ret = [self.tokenizer.bos_id]
        for txt in self.dataset:
            
            ret += txt['text']
            while len(ret) >= self.seq_length:
                tmp = ret[:self.seq_length]
                ret = [self.tokenizer.bos_id] + ret[self.seq_length:]
                yield torch.tensor(tmp)

    def get_stream(self):
        return cycle(self.get_data())

    def __iter__(self):
        return self.get_stream()