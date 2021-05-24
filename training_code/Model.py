import torch
import torch.nn as nn
import torch.nn.functional as F

class ActionPredictor(nn.Module):
    def __init__(self, class_num):
        super().__init__()
        self.linear = nn.Linear(7*7*512, 512)
        class_num = 1 if class_num <= 2 else class_num
        self.output = nn.Linear(512, class_num)
        
    def forward(self, x):
        b, _, _, _ = x.size()
        x = F.relu(self.linear(x.view(b, -1)))
        return self.output(x)
