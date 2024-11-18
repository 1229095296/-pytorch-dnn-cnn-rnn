import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(32, 64, 3),
        #     nn.BatchNorm2d(64),
        #     nn.LeakyReLU(),
        #     nn.MaxPool2d(kernel_size=2)
        # )
        # self.conv3 = nn.Sequential(
        #     nn.Conv2d(64, 128, 3),
        #     nn.BatchNorm2d(128),
        #     nn.LeakyReLU(),
        #     nn.MaxPool2d(kernel_size=2)
        # )
        self.dropout=nn.Dropout(p=0.1)
        self.fc1 = nn.Sequential(
            nn.Linear(111*111*32, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.fc2 = nn.Linear(256, 2)
        
    def forward(self, x):
        x = self.conv1(x)
        # x = self.conv2(x)
        #x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x=self.dropout(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x
