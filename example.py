from dataloaders.wikipedia_dataset import Wikipedia_Dataset
from tokenizers.sptokenizer import SPTokenizer

wp = Wikipedia_Dataset(SPTokenizer())

for i, d in enumerate(wp):
    
    if i > 2:
        break
    print(d.shape)