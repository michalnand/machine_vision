import numpy
import torch

from PIL import Image

import sys
sys.path.append('..')

import Layers

def compute(source_file_name, destination_file_name):

    img     = Image.open(source_file_name)

    img_np  = numpy.array(img)/255.0
    img_np  = img_np.mean(axis=2)


    img_t  = torch.from_numpy(img_np).float()
    img_t  = img_t.unsqueeze(0).unsqueeze(1)

    simple_th   = Layers.Thresholding()
    adaptive_th = Layers.AdaptiveThresholding()

    y_simple   = simple_th(img_t, 0.5)
    y_adaptive = adaptive_th(img_t)


    y_in        = img_np
    y_simple    = y_simple.detach().to("cpu").numpy()[0][0]
    y_adaptive  = y_adaptive.detach().to("cpu").numpy()[0][0]

    y_out       = numpy.hstack([y_in, y_simple, y_adaptive])

    img_y = Image.fromarray(y_out*255)
    img_y.show()
    img_y.convert('RGB').save(destination_file_name)


if __name__ == "__main__":
    compute("../images/text.jpg", "../images/text_thresholding.png")
    compute("../images/computer.jpg", "../images/computer_thresholding.png")