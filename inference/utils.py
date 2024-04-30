
import functools
from datetime import datetime
import numpy as np
import torch

def tensor2img(source:torch.Tensor,imtype=np.uint8):
    image_tensor = source.data
    image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    return image_numpy.astype(imtype)

def get_runtime(f):
    @functools.wraps(f)
    def wrap(*args,**kwargs):
        time_before = datetime.now()
        x = f(*args,**kwargs)
        duration = (datetime.now() - time_before).total_seconds()
        return duration , x
    return wrap