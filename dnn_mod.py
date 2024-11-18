import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class DNN_net1(nn.Module):
    def __init__(self, size=3*224*224):

        super(DNN_net1,self).__init__()
        self.fc1=nn.Linear(size,512)
        # self.fc3=nn.Linear(1024,256)
        # self.fc4=nn.Linear(256,128)
        self.fc5=nn.Linear(512,2)
    def forward(self,x):
        x=x.view(x.size(0),-1)
        x=torch.relu(self.fc1(x))
        # x=torch.relu(self.fc3(x))
        # x=torch.relu(self.fc4(x))
        x=torch.sigmoid(self.fc5(x))
        return x

