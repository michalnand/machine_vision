import torch

class Corners(torch.nn.Module):
    def __init__(self):
        super(Corners, self).__init__() 
        self.blur   = torch.nn.Conv2d(1, 1, kernel_size=5, stride=1, padding=2, bias=False)
        self.blur.weight.data[:,:,:,:] = 1.0/(5.0**2)

        self.conv   = torch.nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1, bias=False)

        kernel = [  [1, 0, -1],
                    [2, 0, -2],
                    [1, 0, -1]]

        self.conv.weight.data[0][0][:][:]   = 1.0/9.0
        self.conv.weight.data[1][0]         = torch.tensor(kernel)
        self.conv.weight.data[2][0]         = torch.tensor(kernel).t()
        
    def forward(self, x):
        x   = self.blur(x)
        fil = self.conv(x)

        level = fil[:,0,:,:].unsqueeze(1)
        gx    = fil[:,1,:,:].unsqueeze(1)
        gy    = fil[:,2,:,:].unsqueeze(1)

        k     = 1.2
        g     = ((gx**2) > k*(level**2)) * ((gy**2) > k*(level**2))
        
        return g

