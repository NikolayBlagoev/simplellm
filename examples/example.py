from simplellm.dataloaders import Tiny_Shakespeare
from simplellm.tokenizers import SPTokenizer
wp = Tiny_Shakespeare(SPTokenizer(),batch_size = 10, seq_l=512)

for i, d in enumerate(wp):
    
    if i > 2:
        break
    print(d.shape)