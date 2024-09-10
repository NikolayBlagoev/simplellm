from simplellm.dataloaders import Tiny_Shakespeare
from simplellm.tokenizers import SPTokenizer
from simplellm.dataloaders import TinyStories
from simplellm.skipllama import LLama,LLamaClassification
from torch import optim, save
import random
import torch.nn.functional as F
seq_l = 64
tkns = SPTokenizer()
wp = TinyStories(tkns,batch_size = 64, seq_l=seq_l)
net = LLama(tkns.vocab_size,dmodel=seq_l*2,num_heads=32,multiple_of=16,ctx_size=seq_l,n_layers=16)

op = optim.Adam(net.parameters())
for i, d in enumerate(wp):
    
    op.zero_grad(set)
    
    x,y = d
    x = x.to("cuda")
    y = y.to("cuda")
    to_skip = random.sample([i for i in range(1,16)],7)
    if i % 10 == 0:
        to_skip = []
    x = net(x,to_skip)
    B, T, C = x.shape
    x = x.view(B*T, C)
    y = y.view(B*T)
    loss = F.cross_entropy(x,y)
    if i % 10 == 0:
        print(loss.item())
    if i%100 == 0:
        save(net.state_dict(), "latest.pth")
    loss.backward()
    op.step()

    