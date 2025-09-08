from simplellm.tokenizers import SPTokenizer
from simplellm.dataloaders import RedPyjama, PretrainDataset, RedPyjamav2
tokenizer = SPTokenizer()
padding_idx = tokenizer.eos_id
train_ds = RedPyjama(tokenizer, batch_size=8, skip = 100, group="default")
train_dl = iter(train_ds)
next(train_dl)
