from itertools import cycle
import torch
from datasets import load_dataset
from simplellm.tokenizers.abstracttokenizer import AbstractTokenizer
from torch.utils.data import DataLoader, IterableDataset
from .pretrain_dataset import PretrainDataset
from ..utils import State
from._abstract_dataset import _AbstractDataset
class RedPyjama(_AbstractDataset):
    

    def __init__(self, tokenizer: AbstractTokenizer, streaming = True, batch_size = 5_000, seq_l=2048, split = 'train',num_workers=0,group = "default", skip = 0, 
                    dataset_type = PretrainDataset):
        if group == "default":
            dataset = []

            for group in ["github","wikipedia","common_crawl","stackexchange","c4"]:
                dataset.append(load_dataset("togethercomputer/RedPajama-Data-1T", group, split=split,streaming = streaming, trust_remote_code=True))
        else:
            dataset = load_dataset("togethercomputer/RedPajama-Data-1T", group, split=split,streaming = streaming, trust_remote_code=True)
        super().__init__(dataset,tokenizer,batch_size,seq_l,num_workers,skip,dataset_type)
    def tokenization(self, t):
        return {"text": self.tokenizer.encode(t["text"])}


