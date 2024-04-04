from typing import Any
import torch

class AbstractTokenizer(object):
    def __init__(self, tokenizer_path: str) -> None:
        self.vocab_size = 0
        self.bos_id: int = 0
        self.eos_id: int = 0
        self.pad_id: int = 0
        pass

    def encode(self, txt: str) -> list[int]:
        return self.sp_model.encode(txt)

    def decode(self, tokens: list[int]) -> str:
        return self.sp_model.decode(tokens)

    def __call__(self, s) -> Any:
        if isinstance(s, str):
            s = [s]
        elif isinstance(s, list):
            s = s
        else:
            raise NotImplemented
        ret = []
        for sent in s:
            ret.append(self.encode(sent))
        return torch.tensor(ret, dtype=torch.long)
        