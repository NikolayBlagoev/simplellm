from simplellm.tokenizers import SPTokenizer
from simplellm.dataloaders import TinyStories, OpenWebText, RedPyjamav2

seq_l = 256
tkns = SPTokenizer()

print(tkns.eos_id)
print(tkns.bos_id)
ts = RedPyjamav2(tkns,batch_size = 1, seq_l=seq_l)
loader = iter(ts) 
while True:
    x = next(loader)
    print(x)
    
    print(x.shape)
    print(tkns.decode(x[0].tolist()))
    print("----")
    print(tkns.decode(x[1].tolist()))
    input()