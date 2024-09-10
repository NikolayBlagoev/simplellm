from simplellm.dataloaders import Tiny_Shakespeare
from simplellm.tokenizers import SPTokenizer
from simplellm.dataloaders import TinyStories
from simplellm.skipllama import LLama,LLamaClassification
from torch import optim, save, no_grad
import random
import torch.nn.functional as F

pth_num = 4
num_to_activate = 8

seq_l = 64
tkns = SPTokenizer()
ts = TinyStories(tkns,batch_size = 64 // pth_num, seq_l=seq_l)
net = LLama(tkns.vocab_size,dmodel=seq_l*2,num_heads=32,multiple_of=16,ctx_size=seq_l,n_layers=16)

op = optim.SGD(net.parameters(),lr=1e-3,momentum=0,dampening=0,weight_decay=0,nesterov=False)
loader = iter(ts)
lr = 1e-3
for i in range(8000//pth_num):
    grad_acc = dict()
    grad_avg = dict()
    loss_hist = []
    op.zero_grad()
    for p in range(pth_num):
        
        x,y = next(loader)
        x = x.to("cuda")
        y = y.to("cuda")
        to_skip = random.sample([i for i in range(1,16)],15-num_to_activate)
        if i % 10 == 0:
            to_skip = []
        x = net(x,to_skip)
        B, T, C = x.shape
        x = x.view(B*T,C)
        y = y.view(B*T)
        loss = F.cross_entropy(x,y)
        if i % 10 == 0:
            loss_hist.append(loss.item())
        loss.backward()
        
        
    op.step()
    if i%10 == 0:
        print(sum(loss_hist)/len(loss_hist))
    if i%100 == 0:
        save(net.state_dict(), "latest.pth")

    
    
    
    
   
    

    