from itertools import cycle
import torch
from datasets import load_dataset
from simplellm.tokenizers.abstracttokenizer import AbstractTokenizer
from torch.utils.data import DataLoader, IterableDataset

class Wikipedia_Dataset(IterableDataset):
    

    def __init__(self, tokenizer: AbstractTokenizer, streaming = True, batch_size = 5_000, seq_l=2048):
        dataset = load_dataset("wikipedia", "20220301.en", split='train', streaming = streaming, trust_remote_code=True)
        #iterable_dataset = dataset.to_iterable_dataset(num_shards=64)
        iterable_dataset = dataset.shuffle(buffer_size=10_000)
        self.batch_size = batch_size
        self.iterable_dataset = iterable_dataset.map(self.tokenization, batched=True, batch_size= batch_size)
        self.dataset =  torch.utils.data.DataLoader(self.iterable_dataset, batch_size= batch_size, collate_fn = self.t)
        # self.dataset = self.dataset.map(self.tokenization, batched=True, batch_size=32)
        self.tokenizer = tokenizer
        self.seq_l = seq_l
        self.padding = [self.tokenizer.pad_id for _ in range(self.seq_l)]
        print("WIKIPEDIA DATASET LOADED...")
    def tokenization(self, t):
        
        #print(t["text"])
        res = self.tokenizer.encode(t["text"])
        
        return {"id": t["id"],"text": res }
    def get_data(self):
        btch = []
        ret = [self.tokenizer.bos_id]
        for sentence in self.iterable_dataset:
            ret += sentence['text']
            #print(sentence)
            while len(ret) >= self.seq_l:
                curr = ret[:self.seq_l]
                ret = [self.tokenizer.bos_id] + ret[self.seq_l:]
                btch.append(curr)
            if len(btch) == self.batch_size:
                yield torch.tensor(btch)
                btch = []

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

    
    def get_stream(self):
        return cycle(self.get_data())
    def __iter__(self):
        return iter(self.dataset)

