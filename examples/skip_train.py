from simplellm.dataloaders import Tiny_Shakespeare
from simplellm.tokenizers import SPTokenizer
from simplellm.dataloaders import TinyStories
from simplellm.llama import LLama,SkipLLama
from torch import optim, save, no_grad
import random
import torch.nn.functional as F

pth_num = 4
num_to_activate = 4

seq_l = 128
tkns = SPTokenizer()
ts = TinyStories(tkns,batch_size = 64 // pth_num, seq_l=seq_l)
net = LLama(SkipLLama,tkns.vocab_size,dmodel=128,num_heads=4,multiple_of=8,ctx_size=seq_l,n_layers=4)

op = optim.SGD(net.parameters(),lr=4e-3/pth_num,momentum=0,dampening=0,weight_decay=0,nesterov=False)

lr = 1e-3
for _ in range(10):
    loader = iter(ts) 
    for i in range(8000):
        grad_acc = dict()
        grad_avg = dict()
        loss_hist = []
        op.zero_grad()
        for p in range(pth_num):
            
            x,y = next(loader)
            x = x.to("cuda")
            y = y.to("cuda")
            if num_to_activate < 4:
                to_skip = random.sample([i for i in range(1,4)],4-num_to_activate)
            else:
                to_skip = []
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
            save(net.state_dict(), f"skip_train_{pth_num}_{num_to_activate}.pth")

    
    
    
    
   
    

    