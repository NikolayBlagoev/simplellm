from sentencepiece import SentencePieceProcessor
from tokenizers.abstracttokenizer import AbstractTokenizer
class SPTokenizer(AbstractTokenizer):
    
    def __init__(self, tokenizer_path: str = "tokenizer.model"):
        self.sp_model = SentencePieceProcessor()
        self.sp_model.load(tokenizer_path)
        self.vocab_size: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        

    def encode(self, txt: str) -> list[int]:
        return self.sp_model.encode(txt)

    def decode(self, tokens: list[int]) -> str:
        return self.sp_model.decode(tokens)