import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset,DataLoader, random_split
from dnn_mod import DNN_net1
from cnn_mod import CNN
from rnn_mod import RNN
import numpy as np
from pathlib import Path
from PIL import Image
from main import get_dataset


device=torch.device("cuda:0"if torch.cuda.is_available() else "cpu")
transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])])

##选择模型进行测试
model=CNN().to(device)
model.load_state_dict(torch.load('CNN.pth', weights_only=True))
# model=RNN().to(device)
# model.load_state_dict(torch.load('RNN.pth', weights_only=True))
# model=DNN_net1().to(device)
# model.load_state_dict(torch.load('DNN.pth', weights_only=True))

test_path="C:\\Users\\Administrator\\Desktop\\data_nlp\\val"
test_data = get_dataset(test_path, transform=transform)
    # print(len(test_data))
loss_function = nn.CrossEntropyLoss()
test_loader = DataLoader(test_data, batch_size=64)
_loss, _corr = 0.0, 0.0
model.eval()
with torch.no_grad():
    for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            y = model(inputs)
            preds = y.argmax(1)
            loss = loss_function(y, labels)
            _loss += loss.item() * inputs.size(0)
            _corr += torch.sum(preds == labels)

    print('Test Loss: {:.4f} Accuracy: {:.4f}%'.format(_loss / len(test_loader.dataset), (_corr / len(test_loader.dataset)) * 100))
                                                          