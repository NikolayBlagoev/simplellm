from itertools import cycle
import torch
from datasets import load_dataset
from simplellm.tokenizers.abstracttokenizer import AbstractTokenizer
from torch.utils.data import DataLoader, IterableDataset
from .pretrain_dataset import PretrainDataset
from ..utils import State
from._abstract_dataset import _AbstractDataset
class GSM(_AbstractDataset):
    

    def __init__(self, tokenizer: AbstractTokenizer, streaming = True, batch_size = 5_000, seq_l=2048, split = 'train',num_workers=0, skip = 0, 
                    pre_prompt = "", dataset_type = PretrainDataset):
        dataset = load_dataset("openai/gsm8k", "main", split=split,streaming = streaming, trust_remote_code=True)
        self.pre_prompt = pre_prompt
        super().__init__(dataset,tokenizer,batch_size,seq_l,num_workers,skip,dataset_type)
    def tokenization(self, t):
        return {"text": self.tokenizer.encode([self.pre_prompt + t["question"][0] + t["answer"][0]])}

    def completion_tokenization(self,t):
        # print(t)
        return {"prompt": self.tokenizer.encode(self.pre_prompt + t["question"][0]), "completion": self.tokenizer.encode(t["answer"][0])}


