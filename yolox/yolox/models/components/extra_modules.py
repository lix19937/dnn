import torch


class Scale(torch.nn.Module):
    def __init__(self, value=1.0):
        super(Scale, self).__init__()
        self.scale = torch.nn.Parameter(torch.tensor(value, dtype=torch.float32))
    
    def forward(self, x):
        return self.scale * x
