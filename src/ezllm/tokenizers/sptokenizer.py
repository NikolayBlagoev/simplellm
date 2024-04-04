from sentencepiece import SentencePieceProcessor
from ezllm.tokenizers.abstracttokenizer import AbstractTokenizer
import requests
import os

class SPTokenizer(AbstractTokenizer):
    
    def __init__(self, tokenizer_path: str = None):
        self.sp_model = SentencePieceProcessor()
        if tokenizer_path != None:
            
            self.sp_model.load(tokenizer_path)
        else:
            
            if not os.path.isfile('tokenizer.model'):
                
                response = requests.get('https://huggingface.co/togethercomputer/LLaMA-2-7B-32K/resolve/main/tokenizer.model?download=true', stream=True)
                with open('tokenizer.model','wb') as output:
                    print("downloading to ", os.path.abspath(output.name))
                    output.write(response.content)
            
            self.sp_model.load('tokenizer.model')
        self.vocab_size: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        

    def encode(self, txt: str) -> list[int]:
        return self.sp_model.encode(txt)

    def decode(self, tokens: list[int]) -> str:
        return self.sp_model.decode(tokens)