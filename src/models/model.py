import torch.nn as nn
import torch
import torch.nn.functional as F

class Paper_network(nn.Module):
    def __init__(self,input_size, dropout):
        super(Paper_network, self).__init__()
        self.layer1 = nn.Linear(input_size, 1024)
        self.layer2 = nn.Linear(1024,512)
        self.layer3 = nn.Linear(512, 256)
        self.layer4 = nn.Linear(256, 128)
        self.layer5 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.layer2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.layer3(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.layer4(x)
        x = F.relu(x)
        x = self.dropout(x)        

        x = self.layer5(x)
        x = F.relu(x)
        x = self.dropout(x)    

        x = self.output(x)

        x = torch.sigmoid(x)
        return x