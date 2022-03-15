import torch

class AdaptiveThresholding(torch.nn.Module):
    def __init__(self, kernel_size = 7):
        super(AdaptiveThresholding, self).__init__()
        self.conv   = torch.nn.Conv2d(1, 1, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=False)

        self.conv.weight.data[0][0][:][:] = 1.0/(kernel_size**2)
        
        
    def forward(self, x, level = 0.9):
        intensity = self.conv(x)
        y = 1.0*(x > intensity*level)

        return y

