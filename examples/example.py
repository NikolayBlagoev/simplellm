from ezllm.dataloaders import Wikipedia_Dataset
from ezllm.tokenizers import SPTokenizer
wp = Wikipedia_Dataset(SPTokenizer(),batch_size = 500, seq_l=512)

for i, d in enumerate(wp):
    
    if i > 2:
        break
    print(d.shape)