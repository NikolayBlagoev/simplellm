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
            
            tmp = [self.tokenizer.bos_id] + txt['text']
            
            while len(tmp) >= self.seq_length + 1:
                tmp_x = tmp[:self.seq_length]
                tmp =  tmp[self.seq_length:]
               
                yield torch.tensor(tmp_x)
            if len(tmp) < self.seq_length // 2:
                
                continue
            tmp += [self.tokenizer.eos_id]
            
            ret = torch.nn.functional.pad(torch.tensor(tmp),(0,self.seq_length - len(tmp)),value=self.tokenizer.pad_id)

            yield ret

    def get_stream(self):
        return cycle(self.get_data())

    def __iter__(self):
        return self.get_stream()