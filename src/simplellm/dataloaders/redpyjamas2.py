from itertools import cycle
import torch
from datasets import load_dataset
from simplellm.tokenizers.abstracttokenizer import AbstractTokenizer
from torch.utils.data import DataLoader, IterableDataset
from .pretrain_dataset import PretrainDataset
from ..utils import State
from._abstract_dataset import _AbstractDataset
class RedPyjamav2(_AbstractDataset):
    

    def __init__(self, tokenizer: AbstractTokenizer, streaming = True, batch_size = 5_000, seq_l=2048, split = 'train',num_workers=0,name = "default", skip = 0, 
                    dataset_type = PretrainDataset):
        dataset = load_dataset("togethercomputer/RedPajama-Data-V2", name=name, streaming = streaming, trust_remote_code=True,languages=["en"], split="train")
        
        super().__init__(dataset,tokenizer,batch_size,seq_l,num_workers,skip,dataset_type)
    def tokenization(self, t):
        
        return {"text": list(map(lambda el: self.tokenizer.encode(el),t["raw_content"]))}


