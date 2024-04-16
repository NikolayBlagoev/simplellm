from sentencepiece import SentencePieceProcessor
from simplellm.tokenizers.abstracttokenizer import AbstractTokenizer
import requests
import os

class SPTokenizer(AbstractTokenizer):
    
    def __init__(self, tokenizer_path: str = None):
        self.sp_model = SentencePieceProcessor()
        if tokenizer_path != None:
            
            self.sp_model.load(tokenizer_path)
        else:
            
            if not os.path.isfile('llama-tokenizer.model'):
                
                response = requests.get('https://huggingface.co/togethercomputer/LLaMA-2-7B-32K/resolve/main/tokenizer.model?download=true', stream=True)
                with open('llama-tokenizer.model','wb') as output:
                    print("downloading to ", os.path.abspath(output.name))
                    output.write(response.content)
            
            self.sp_model.load('llama-tokenizer.model')
        self.vocab_size: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = 0
        

    def encode(self, txt: str) -> list[int]:
        return self.sp_model.encode(txt)

    def decode(self, tokens: list[int]) -> str:
        return self.sp_model.decode(tokens)