import torch
import torch.nn as nn

device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
class TestModelGPU(nn.Module):
    def __init__(self,nodes,output):
        super(TestModelGPU, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.linear = nn.Linear(nodes, output).to(self.device)

    def forward(self, x):
        return self.linear(x)

class TestModel(nn.Module):
    def __init__(self,nodes):
        super(TestModelGPU, self).__init__()
       # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.linear = nn.Linear(nodes, 3)

    def forward(self, x):
        return self.linear(x)

