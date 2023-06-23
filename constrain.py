###
import torch
import torch.nn as nn


def one_hot(depth,value):
    ''''
    one hot 编码
    '''
    one=torch.eye(depth)[value]
    return one

class negtive_log(nn.Module):
    def __init__(self):
        super(negtive_log, self).__init__()
    def forward(self, x,y):
        ###x[B,N] y[B,c]
        y=one_hot(depth=x.size(1),value=y)
        #y=torch.from_numpy(y)
        positive=torch.eq(y,1)
        negtive=torch.eq(y,0)
        label_t=torch.masked_select(x,positive).view(-1,1)
        #print("label_t ",label_t.size())
        label_f=torch.masked_select(x,negtive).view(x.size(0),-1)
        #print("label_f",label_f.size())
        print("样本值", label_t)
        label_positive=torch.log(label_t)
        print("average", label_f)
        label_negative=torch.log(1.0-label_f)
        average_negative=torch.mean(label_negative,dim=-1,keepdim=True)
        #print("average_negative",average_negative.size())
        #print("average",average_negative)
        loss_1=torch.mean(label_positive+average_negative,dim=0)
        loss_1=0.0-loss_1
        #rint("里面loss",loss_1.size())
        return loss_1
