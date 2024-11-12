from itertools import cycle
import torch
from datasets import load_dataset
from simplellm.tokenizers.abstracttokenizer import AbstractTokenizer
from torch.utils.data import DataLoader, IterableDataset

class Wikipedia_Dataset(object):
    

    def __init__(self, tokenizer: AbstractTokenizer, streaming = True, batch_size = 5_000, seq_l=2048, split = 'train',num_workers=0):
        dataset = load_dataset("wikipedia", "20220301.en", split=split, streaming = streaming, trust_remote_code=True)
        
        iterable_dataset = dataset.shuffle(buffer_size=10_000)

        self.batch_size = batch_size
        self.iterable_dataset = AbstractDataset(iterable_dataset, tokenizer, seq_l)

        self.dl = torch.utils.data.DataLoader(self.iterable_dataset,batch_size=batch_size,shuffle=False, num_workers=num_workers,pin_memory=True,collate_fn=None)
        self.tokenizer = tokenizer
        self.seq_l = seq_l
        print("WIKIPEDIA DATASET LOADED...")
    
    def get_data(self):
        return self.dl
    
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