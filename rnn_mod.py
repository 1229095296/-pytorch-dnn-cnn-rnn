import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
class RNN(nn.Module):
    def __init__(self):
        super(RNN,self).__init__()
        self.rnn=nn.RNN(input_size=224*3,hidden_size=256,nonlinearity='relu',num_layers=2,batch_first=True)
        self.fc=nn.Linear(256,2)
    def forward(self,x):
        x=x.view(x.size(0),224,224*3)#把x展平为适合rnn的数据格式，batchsize,seqlength,inputsize
        out,h_n=self.rnn(x)
        out=self.fc(out[:,-1,:])#out的形状是batchsize，seqlength，hiddensize，-1表示最后一时间步
        return out