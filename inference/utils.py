
import functools
from datetime import datetime
import os
import PIL
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def tensor2img(source:torch.Tensor,imtype=np.uint8):
    image_tensor = source.data
    image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    return image_numpy.astype(imtype)

def get_transform(name=None):
    return transforms.Compose([
        transforms.Resize((256, 256), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])  

def get_input_image(source) -> PIL.Image:
    input_image = None
    if isinstance(source, str):
        input_image = Image.open(source).convert('RGB')
    elif isinstance(source, np.ndarray):
        input_image = Image.fromarray(source).convert('RGB')
    elif  isinstance(source,PIL.PngImagePlugin.PngImageFile):
        input_image = source
    else:
        raise Exception('Unsupported type')
    return input_image

def get_runtime(f):
    @functools.wraps(f)
    def wrap(*args,**kwargs):
        time_before = datetime.now()
        x = f(*args,**kwargs)
        duration = (datetime.now() - time_before).total_seconds()
        return duration , x
    return wrap