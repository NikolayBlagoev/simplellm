from simplellm.dataloaders import Tiny_Shakespeare
from simplellm.tokenizers import SPTokenizer
from simplellm.dataloaders import TinyStories
from simplellm.swapllama import LLama,LLamaClassification
from torch import optim, save, no_grad
import random
import torch.nn.functional as F

pth_num = 4
num_to_switch = 1

seq_l = 128
tkns = SPTokenizer()
ts = TinyStories(tkns,batch_size = 64 // pth_num, seq_l=seq_l)
net = LLama(tkns.vocab_size,dmodel=256,num_heads=8,multiple_of=256,ctx_size=seq_l,n_layers=16)

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
            swaps = [0]
            exec_order = [i for i in range(0,16)]
            if i % 10 != 0:
                for _ in range(num_to_switch):
            
                    swp_idx = 0
                    while swp_idx in swaps or swp_idx + 1 in swaps:
                        swp_idx=random.randint(1,14)
                
                    exec_order[swp_idx] = swp_idx + 1
                    exec_order[swp_idx+1] = swp_idx 
                    swaps.append(swp_idx)
                    swaps.append(swp_idx+1)
            
            x = net(x,exec_order)
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
            save(net.state_dict(), f"swap_train_{pth_num}_{num_to_switch}.pth")

    
    
    
    
   
    

    