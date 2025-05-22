from transformers import AutoModelForCausalLM, AutoTokenizer
from simplellm.tokenizers.abstracttokenizer import AbstractTokenizer


class HFTokenizer(AbstractTokenizer):
    
    def __init__(self, tokenizer_name):
        self.tkns = AutoTokenizer.from_pretrained(tokenizer_name)
        self.vocab_size: int = self.tkns.vocab_size
        self.bos_id: int = self.tkns(self.tkns.bos_token).input_ids[0]
        self.eos_id: int = self.tkns(self.tkns.eos_token).input_ids[0]
        self.pad_id: int = 0
        
        

    def encode(self, txt: str) -> list[int]:
        return self.tkns(txt).input_ids

    def decode(self, tokens: list[int]) -> str:
        return self.tkns.decode(tokens)