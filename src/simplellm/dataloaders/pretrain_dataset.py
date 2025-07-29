from itertools import cycle
import torch
from datasets import load_dataset
from simplellm.tokenizers.abstracttokenizer import AbstractTokenizer
from torch.utils.data import DataLoader, IterableDataset

class PretrainDataset(IterableDataset):
    def __init__(self, dataset, tokenizer, seq_length=2048):
        if isinstance(dataset, list):

            self.dataset = dataset
            self.holders = [1 for _ in range(len(self.dataset))]
        else:
            self.dataset = [dataset]
            self.holders = [1]

        self.tokenizer = tokenizer
        self.seq_length = seq_length
        

    def get_data(self):
        tmp = []
        curr_ds = 0
        tmp_data = [iter(dataset) for dataset in self.dataset]
        self.holders = [1 for _ in range(len(self.dataset))]
        
        while sum(self.holders) > 0:
            # print("getting from",curr_ds)
            try:
                txt = next(tmp_data[curr_ds])
                tmp += txt['text'] + [self.tokenizer.eos_id]
                # print("from", curr_ds)
                while len(tmp) >= self.seq_length:
                    tmp_x = tmp[:self.seq_length]
                    tmp =  tmp[self.seq_length:]
                    to_yield = torch.tensor(tmp_x)
                    yield to_yield
            except StopIteration:
                self.holders[curr_ds] = 0
            
            if sum(self.holders) == 0:
                break
            curr_ds += 1
            curr_ds = curr_ds % len(self.holders)
            while self.holders[curr_ds] == 0:
                curr_ds += 1   
                curr_ds = curr_ds % len(self.holders)         

    def get_stream(self):
        return cycle(self.get_data())

    def __iter__(self):
        return self.get_stream()