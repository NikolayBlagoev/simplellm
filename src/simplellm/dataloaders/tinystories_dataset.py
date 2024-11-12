from itertools import cycle
import torch
from datasets import load_dataset
from .abstract_dataset import AbstractDataset
class TinyStories(IterableDataset):
    

    def __init__(self, tokenizer: AbstractTokenizer, streaming = True, batch_size = 5_000, seq_l=2048, split = 'train', num_workers = 1):
        dataset = load_dataset("roneneldan/TinyStories", split=split, streaming = streaming, trust_remote_code=True)
        iterable_dataset = dataset.shuffle(buffer_size=10_000)

        self.batch_size = batch_size
        self.iterable_dataset = AbstractDataset(iterable_dataset, tokenizer, seq_l)

        self.dl = torch.utils.data.DataLoader(self.iterable_dataset,batch_size=batch_size,shuffle=False, num_workers=num_workers,pin_memory=True,collate_fn=None)
        self.tokenizer = tokenizer
        self.seq_l = seq_l
        print("TINYSTORIES DATASET LOADED...")
    
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

