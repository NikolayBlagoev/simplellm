from itertools import cycle
import torch
from datasets import load_dataset
from simplellm.tokenizers.abstracttokenizer import AbstractTokenizer
from torch.utils.data import DataLoader, IterableDataset

class TinyStories(IterableDataset):
    

    def __init__(self, tokenizer: AbstractTokenizer, streaming = True, batch_size = 5_000, seq_l=2048, split = 'train'):
        dataset = load_dataset("roneneldan/TinyStories", split=split, streaming = streaming, trust_remote_code=True)
        
        iterable_dataset = dataset.shuffle(buffer_size=10_000)
        self.batch_size = batch_size
        self.iterable_dataset = iterable_dataset.map(self.tokenization, batched=True, batch_size=batch_size)
        self.dataset =  torch.utils.data.DataLoader(self.iterable_dataset, batch_size= batch_size, collate_fn = self.t)
        
        self.tokenizer = tokenizer
        self.seq_l = seq_l
        self.padding = [self.tokenizer.pad_id for _ in range(self.seq_l)]
        print("TINYSTORIES DATASET LOADED...")
    def tokenization(self, t):
        
        #print(t["text"])
        
        
        return {"text": self.tokenizer.encode(t["text"])}
    def get_data(self):
        print("getting data...")
        
        
        btch = []
        ret = [self.tokenizer.bos_id]
        target = []
        trgt_btch = []
        for sentence in self.iterable_dataset:
            
            ret += sentence["text"]
            target += sentence["text"]
            #print(sentence)
            while len(ret) >= self.seq_l + 1:
                curr = ret[:self.seq_l]
                curr2 = target[:self.seq_l]
                ret = ret[1:]
                target = target[1:]
                btch.append(curr)
                
                trgt_btch.append(curr2)
                
                if len(btch) == self.batch_size:
                    yield torch.tensor(btch), torch.tensor(trgt_btch)
                    btch = []
                    trgt_btch = []
        self.iterable_dataset = iterable_dataset.map(self.tokenization, batched=True, batch_size=batch_size)
        self.dataset =  torch.utils.data.DataLoader(self.iterable_dataset, batch_size= batch_size, collate_fn = self.t)

    def t(self, i):
        
        res = [t["text"] for t in i]
        trgt = []
        
        max_l = self.seq_l
        for i in range(len(res)):
            if len(res[i]) < max_l:
                res[i] += self.padding[:max_l-len(res[i])]
                
            else:
                res[i] = res[i][:self.seq_l]
            trgt.append(res[i][1:])
        
        return torch.tensor(res),torch.tensor(trgt)
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

