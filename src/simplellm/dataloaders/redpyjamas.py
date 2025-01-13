from itertools import cycle
import torch
from datasets import load_dataset
from simplellm.tokenizers.abstracttokenizer import AbstractTokenizer
from .abstract_dataset import AbstractDataset
class RedPyjama(object):
    

    def __init__(self, tokenizer: AbstractTokenizer, batch_size = 5_000, seq_l=2048,num_workers=0):
        dataset = load_dataset("togethercomputer/RedPajama-Data-1T", "c4", trust_remote_code=True,streaming=True)
        iterable_dataset = dataset.shuffle(buffer_size=10_000)
        iterable_dataset = iterable_dataset.map(self.tokenization, batched=True, batch_size=batch_size)
        self.batch_size = batch_size
        self.iterable_dataset = AbstractDataset(iterable_dataset, tokenizer, seq_l)

        self.dl = torch.utils.data.DataLoader(self.iterable_dataset,batch_size=batch_size,shuffle=False, num_workers=num_workers,pin_memory=True,collate_fn=None)
        self.tokenizer = tokenizer
        self.seq_l = seq_l
        print("RedPajama DATASET LOADED...")
    
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

