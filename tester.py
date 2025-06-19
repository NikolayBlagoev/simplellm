from simplellm.tokenizers import SPTokenizer
from simplellm.dataloaders import OpenWebText,TinyStories


sp = SPTokenizer()
ds = TinyStories(sp,batch_size=2)
itr = iter(ds)
print(next(itr))