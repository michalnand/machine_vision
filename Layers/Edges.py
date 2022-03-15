import torch

class Edges(torch.nn.Module):
    def __init__(self):
        super(Edges, self).__init__()
        self.blur        = torch.nn.Conv2d(1, 1, kernel_size=5, stride=1, padding=2, bias=False)
        self.blur.weight.data[:,:,:,:] = 1.0/(5.0**2)

        self.conv_edges  = torch.nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1, bias=False)

        kernel = [  [1, 0, -1],
                    [2, 0, -2],
                    [1, 0, -1]]

        self.conv_edges.weight.data[0][0] = torch.tensor(kernel)
        self.conv_edges.weight.data[1][0] = torch.tensor(kernel).t()
        
    def forward(self, x):
        x     = self.blur(x)
        edges = self.conv_edges(x)

        gx    = edges[:,0,:,:].unsqueeze(1)
        gy    = edges[:,1,:,:].unsqueeze(1)

        g     = ((gx**2) + (gy**2))**0.5
        
        return g

