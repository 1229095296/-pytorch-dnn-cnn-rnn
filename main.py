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
import matplotlib.pyplot as plt
transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])])
device=torch.device("cuda:0"if torch.cuda.is_available() else "cpu")

get_label=lambda x:x.name.split('.')[0]
class get_dataset(Dataset):
    def __init__(self, root, transform=None):
        self.images=list(Path(root).glob('*.jpg'))
        self.transform=transform
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        img=self.images[idx]
        label=get_label(img)
        label=1 if label=='dog' else 0
        if self.transform:
            img=self.transform(Image.open(img))
        return img, torch.tensor(label,dtype=torch.int64)
if __name__ == '__main__':
    datasets=get_dataset(root="C:\\Users\\Administrator\\Desktop\\data_nlp\\train",transform=transform) 
    train_data,valid_data=random_split(datasets,lengths=[int(len(datasets)*0.8),int(len(datasets)*0.2)],generator=torch.Generator().manual_seed(7))
    train_loader=DataLoader(train_data,batch_size=64,shuffle=True)
    valid_loader=DataLoader(valid_data,batch_size=64)

#选择模型
    #model=DNN_net1().to(device)
    #model=CNN().to(device)
    model=RNN().to(device)

#交叉熵损失函数定义
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3,weight_decay=1e-4)##尝试调整学习率以改变测试结果

    epochs = 25
    train_loss_list = []
    train_acc_list = []
    for epoch in range(epochs):
        print("Epoch {} / {}".format(epoch + 1, epochs))
                
        t_loss, t_corr = 0.0, 0.0
        
        model.train() 
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            preds = model(inputs)
            loss = loss_function(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t_loss += loss.item() * inputs.size(0)
            t_corr += torch.sum(preds.argmax(1) == labels) 
            # preds.argmax(1)返回预测结果中概率最大的类别标签，即预测的类别
            
        train_loss = t_loss / len(train_loader.dataset)
        train_acc = t_corr.cpu().numpy() / len(train_loader.dataset)#将张量tcorr移动到cpu上以支持numpy库
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)  
        print('Train Loss: {:.4f} Accuracy: {:.4f}%'.format(train_loss, train_acc * 100))
        
        #torch.save(model.state_dict(), 'CNN.pth')  # 保存模型
        #torch.save(model.state_dict(), 'RNN.pth')  # 保存模型
        #torch.save(model.state_dict(), 'DNN.pth')  # 保存模型
    plt.figure()
    plt.title('Train Loss and Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('')
    plt.plot(range(1, epochs+1), np.array(train_loss_list), color='blue',
            linestyle='-', label='Train_Loss')
    plt.plot(range(1, epochs+1), np.array(train_acc_list), color='red',
            linestyle='-', label='Train_Accuracy')
    plt.legend()  # 凡例
    plt.savefig('rnn_train.png')
    plt.show()  # 表示
    ##绘制训练趋势图部分


