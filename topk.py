import torch
import torch.nn as nn

output = torch.FloatTensor([[0.91,0,0.9,0,0],
							[1.0,0,0.98,0,0.9],
							[0,0,0.9,1.0,0],
							[1.0,0,1.0,0,0],
							[0.9,1.0,0.98,0.89,0.60]])
topk = (1,)
maxk = max(topk)
_, pred = output.topk(maxk,1,True,True)
print("pred:",pred)
print("_:",_)
pred = pred.t()
print("pred:",pred)
target = torch.FloatTensor([[0, 0, 2, 2, 1]])
print("target:",target.view(1, -1))
exp_t = target.view(1, -1).expand_as(pred)
print("exp_t:",exp_t)
correct = pred.eq(exp_t.long())
print("correct:",correct)
for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    print("correct_k:",correct_k)
    top1_acc = correct_k.mul_(100.0 / 5)
    print("top1_acc:",top1_acc)
