import numpy
from PIL import Image

import torch

class HOGLayer(torch.nn.Module):
    def __init__(self, nbins=10, pool=8):
        super(HOGLayer, self).__init__()

        self.nbins  = nbins

        self.conv   = torch.nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1, bias=False)

        kernel = [  [1, 0, -1],
                    [2, 0, -2],
                    [1, 0, -1]]

        self.conv.weight.data[0][0] = torch.tensor(kernel)
        self.conv.weight.data[1][0] = torch.tensor(kernel).t()


        self.pooling = torch.nn.AvgPool2d(pool, stride=pool)

    def forward(self, x):

        gxy = self.conv(x)

        phase = torch.atan2(gxy[:,0,:,:], gxy[:,1,:,:])

        phase_int = phase / torch.pi * self.nbins
        phase_int = phase_int[:,None,:,:]

        '''
        #2. Mag/ Phase
        mag = gxy.norm(dim=1)
        norm = mag[:,None,:,:]

        #3. Binning Mag with linear interpolation
        phase_int = phase / self.max_angle * self.nbins
        phase_int = phase_int[:,None,:,:]

        n, c, h, w = gxy.shape
        out = torch.zeros((n, self.nbins, h, w), dtype=torch.float, device=gxy.device)
        out.scatter_(1, phase_int.floor().long()%self.nbins, norm)
        out.scatter_add_(1, phase_int.ceil().long()%self.nbins, 1 - norm)
        '''

        return phase



img = Image.open("../images/computer.jpg")

img_np = numpy.array(img)/255.0

img_t  = torch.from_numpy(img_np).float()
img_t  = torch.mean(img_t, dim=2).unsqueeze(0).unsqueeze(1)

hog = HOGLayer()

y = hog(img_t)

y_np = y.detach().to("cpu").numpy()[0]

print(img_t.shape, y_np.shape)

y_np  = (y_np + numpy.pi)/(2.0*numpy.pi)
img_y = Image.fromarray(y_np*255)
img_y.show()

