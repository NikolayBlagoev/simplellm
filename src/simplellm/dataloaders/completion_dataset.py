from itertools import cycle
import torch
from datasets import load_dataset
from simplellm.tokenizers.abstracttokenizer import AbstractTokenizer
from torch.utils.data import DataLoader, IterableDataset

class CompletionDataset(IterableDataset):
    def __init__(self, dataset, tokenizer: AbstractTokenizer, seq_length=2048):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.seq_length = seq_length

    def get_data(self):
        
        for txt in self.dataset:
           
            if self.tokenizer.bos_id != None:

                x = [self.tokenizer.bos_id]
            else:
                x = []
            x += txt['prompt']
            y = txt['completion']
            print(x,y)
            if "additional" not in txt:
                yield {"prompt": [torch.tensor(x)], "completion": [torch.tensor(y)]}
            else:
                yield {"prompt": [x], "completion": [y], "additional": [txt["additional"]]}

            
            

    def get_stream(self):
        return cycle(self.get_data())

    def __iter__(self):
        return self.get_stream()