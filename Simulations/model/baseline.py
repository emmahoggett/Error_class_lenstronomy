import torch
import torch.nn as nn
import torch.nn.functional as F



class CNNNetBasic(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 13 * 13, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)
        self.sigmoid = nn.Sigmoid()
        self.typenet = 'conv'
        

    def forward(self, x: torch.Tensor):
        x = self.pool(F.selu(self.conv1(x)))
        x = self.pool(F.selu(self.conv2(x)))
        x = x.view(x.size(0), 16 * 13 * 13)
        x = F.selu(self.fc1(x))
        x = F.selu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x
        
class TabularNetBasic(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(11, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 3)
        self.typenet = 'meta'


    def forward(self, x: torch.Tensor):

        x = F.selu(self.fc1(x))
        x = F.selu(self.fc2(x))
        x = nn.Sigmoid(self.fc3(x))

        return x

class TabularCNNNetBasic(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1_data = nn.Linear(11, 16)
        self.fc2_data = nn.Linear(16, 8)

        self.conv1_img = nn.Conv2d(1, 6, kernel_size=5)
        self.pool_img = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2_img = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1_img = nn.Linear(16 * 13 * 13, 120)
        self.fc2_img = nn.Linear(120, 84)

        self.fc1 = nn.Linear(8 + 84, 60)
        self.fc2 = nn.Linear(60, 3)
        self.typenet = 'convXmeta'
        

    def forward(self, img: torch.Tensor, data: torch.Tensor):

        data = F.selu(self.fc1_data(data))
        data = F.selu(self.fc2_data(data))

        img = self.pool_img(F.selu(self.conv1_img(img)))
        img = self.pool_img(F.selu(self.conv2_img(img)))
        img = img.view(img.size(0), 16 * 13 * 13)
        img = F.selu(self.fc1_img(img))
        img = F.selu(self.fc2_img(img))

        x = torch.cat((img, data), dim=1)
        x = F.relu(self.fc1(x))
        x = nn.Sigmoid(self.fc2(x))

        return x
        



