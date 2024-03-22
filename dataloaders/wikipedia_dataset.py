from itertools import cycle
import torch
from datasets import load_dataset
from tokenizers.abstracttokenizer import AbstractTokenizer
from torch.utils.data import DataLoader, IterableDataset
def gen_dataset():
    dataset = load_dataset("wikipedia")
    iterable_dataset = dataset.to_iterable_dataset(num_shards=64) # shard the dataset
    iterable_dataset = iterable_dataset.shuffle(buffer_size=10_000)  # shuffles the shards order and use a shuffle buffer when you start iterating
    return torch.utils.data.DataLoader(iterable_dataset)

class Wikipedia_Dataset(IterableDataset):
    def __init__(self, tokenizer: AbstractTokenizer, seq_l=2048):
        dataset = load_dataset("wikipedia")
        iterable_dataset = dataset.to_iterable_dataset(num_shards=64)
        iterable_dataset = iterable_dataset.shuffle(buffer_size=10_000)
        self.dataset = iterable_dataset
        self.tokenizer = tokenizer
        self.seq_l = seq_l
    def get_data(self):
        ret = [self.tokenizer.bos_id]
        for sentence in self.dataset:
            tokenized = self.tokenizer.encode(sentence)
            tokenized.append(self.tokenizer.eos_id)
            ret += tokenized
            while len(ret) >= self.seq_l:
                curr = ret[:self.seq_l]
                ret = [self.tokenizer.bos_id] + ret[self.seq_l:]
                yield torch.tensor(curr)
            yield torch.tensor(ret)
            ret = [self.tokenizer.bos_id]
    def __iter__(self):
        return cycle(self.get_data())
