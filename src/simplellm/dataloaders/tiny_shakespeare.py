from itertools import cycle
import torch
from datasets import load_dataset
from simplellm.tokenizers.abstracttokenizer import AbstractTokenizer
from torch.utils.data import DataLoader, IterableDataset

class Tiny_Shakespeare(IterableDataset):
    

    def __init__(self, tokenizer: AbstractTokenizer, batch_size = 5_000, seq_l=2048):
        dataset = load_dataset("tiny_shakespeare")["train"]
        #iterable_dataset = dataset.to_iterable_dataset(num_shards=64)
        iterable_dataset = dataset
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.iterable_dataset = iterable_dataset.map(self.tokenization, batched=True, batch_size=batch_size)
        self.iterable_dataset = torch.tensor(self.iterable_dataset["text"], dtype=torch.long).squeeze()
        
        self.ln = self.iterable_dataset.shape[0]
        print(self.iterable_dataset.shape[0])
        self.seq_l = seq_l
        self.padding = [self.tokenizer.pad_id for _ in range(self.seq_l)]
        print("TINY SHAKESPEARE DATASET LOADED...")
    def tokenization(self, t):
        
        #print(t["text"])
        # print(t)
        res = self.tokenizer.encode(t["text"])
        
        return {"text": res }
    def get_data(self):
        print("getting data")
        i = 0
        while i < self.ln:
            
            loc_batch_size  = min(self.batch_size, (self.ln - i)//self.seq_l)
            # for k in range(loc_batch_size):
            #     print(self.iterable_dataset[i+k*self.seq_l:i+ k*self.seq_l+self.seq_l].shape, i+k*self.seq_l,i+ k*self.seq_l+self.seq_l)
            yield torch.stack([self.iterable_dataset[i+k*self.seq_l:i+ k*self.seq_l+self.seq_l] for k in range(loc_batch_size)])
            i += self.seq_l*self.batch_size
        raise StopIteration
    def t(self, i):
        res = [t["text"] for t in i]
        max_l = 0
        for i in range(len(res)):
            max_l = max(len(res[i]), max_l)
        
        max_l = min(max_l, self.seq_l)
        for i in range(len(res)):
            if len(res[i]) < max_l:
                res[i] += self.padding[:max_l-len(res[i])]
            else:
                res[i] = res[i][:self.seq_l]
            
        return torch.tensor(res)
    def decode(self, tnsrs: torch.Tensor) -> list[str]:
        b, _ = tnsrs.shape
        ret = []
        for t in tnsrs:
            ret.append(self.tokenizer.decode(t.tolist()))
        return ret

    
    def get_stream(self):
        return cycle(self.get_data())
    def __iter__(self):
        return cycle(self.get_data())

