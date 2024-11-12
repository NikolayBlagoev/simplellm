from simplellm.tokenizers import SPTokenizer
from simplellm.dataloaders import TinyStories

seq_l = 2048
tkns = SPTokenizer()
ts = TinyStories(tkns,batch_size = 64, seq_l=seq_l)
loader = iter(ts) 
while True:
    x,y = next(loader)
    print(x.shape)