import torch
import torch.nn as nn

device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
class TestModel(nn.Module):
    def __init__(self,nodes,output,device = True):
        super(TestModel, self).__init__()
        if not device: 
            self.device = "cpu"
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.linear = nn.Linear(nodes, output).to(self.device)

    def forward(self, x):
        return self.linear(x)

