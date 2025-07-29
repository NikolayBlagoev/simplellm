from itertools import cycle
import torch
from datasets import load_dataset
from simplellm.tokenizers.abstracttokenizer import AbstractTokenizer
from torch.utils.data import DataLoader, IterableDataset
from .pretrain_dataset import PretrainDataset
from .completion_dataset import CompletionDataset
from ..utils import State
from abc import abstractmethod
class _AbstractDataset(object):
    

    def __init__(self, dataset, tokenizer: AbstractTokenizer, batch_size = 5_000, seq_l=2048, num_workers=0, skip = 0, dataset_type = PretrainDataset):
        if isinstance(dataset, list):
            iterable_dataset = [d.shuffle(buffer_size=10_000, seed=State.get_seed()).skip(skip) for d in dataset]
        else:
            iterable_dataset = dataset.shuffle(buffer_size=10_000, seed=State.get_seed()).skip(skip)
        
        self.batch_size = batch_size
        if dataset_type == CompletionDataset:
            if isinstance(iterable_dataset, list):
                iterable_dataset = [d.map(self.completion_tokenization, batched=True, batch_size=batch_size) for d in iterable_dataset]
            else:
                iterable_dataset = iterable_dataset.map(self.completion_tokenization, batched=True, batch_size=batch_size)
        else:
            if isinstance(iterable_dataset, list):
                iterable_dataset = [d.map(self.tokenization, batched=True, batch_size=batch_size) for d in iterable_dataset]
            else:
                iterable_dataset = iterable_dataset.map(self.tokenization, batched=True, batch_size=batch_size)
        self.iterable_dataset = dataset_type(iterable_dataset, tokenizer, seq_l)

        self.dl = torch.utils.data.DataLoader(self.iterable_dataset,batch_size=batch_size,shuffle=False, num_workers=num_workers,pin_memory=True,collate_fn=None)
        self.tokenizer = tokenizer
        self.seq_l = seq_l
        print(f".... DATASET LOADED...")
    @abstractmethod
    def tokenization(self, t):
        raise NotImplementedError(self.__class__,"has no implemented tokenization metod")
    
    @abstractmethod
    def completion_tokenization(self, t):
        raise NotImplementedError(self.__class__,"has no implemented completion_tokenization metod")
    
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


