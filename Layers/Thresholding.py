import torch

class Thresholding(torch.nn.Module):
    def __init__(self):
        super(Thresholding, self).__init__()
       
    def forward(self, x, level = 0.98):
        y = 1.0*(x > level)

        return y

