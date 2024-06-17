
import functools
from datetime import datetime
import json
import os
import pdb
from skimage import color
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

def get_transform(name=None,colorization=False):
    trans = []
    if colorization:
        trans+= [transforms.Grayscale(1)]
    trans += [
        transforms.Resize((256, 256), Image.BICUBIC),
        transforms.ToTensor(),
        ]
    if colorization:
        trans+=[transforms.Normalize((0.5), (0.5))]
    else:
        trans+=[transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(trans)  

def get_input_image(source) -> PIL.Image:
    input_image = None
    if isinstance(source, str):
        input_image = Image.open(source).convert('RGB')
    elif isinstance(source, np.ndarray):
        input_image = Image.fromarray(source).convert('RGB')
    elif  isinstance(source,PIL.PngImagePlugin.PngImageFile) or isinstance(source,PIL.JpegImagePlugin.JpegImageFile):
        input_image = source.convert('RGB')
    else:
        # pdb.set_trace()
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

def lab2rgb(L, AB):
    """Convert an Lab tensor image to a RGB numpy output
    Parameters:
        L  (1-channel tensor array): L channel images (range: [-1, 1], torch tensor array)
        AB (2-channel tensor array):  ab channel images (range: [-1, 1], torch tensor array)

    Returns:
        rgb (RGB numpy image): rgb output images  (range: [0, 255], numpy array)
    """
    AB2 = AB * 110.0
    L2 = (L + 1.0) * 50.0
    Lab = torch.cat([L2, AB2], dim=1)
    Lab = Lab[0].data.cpu().float().numpy()
    Lab = np.transpose(Lab.astype(np.float64), (1, 2, 0))
    rgb = color.lab2rgb(Lab)
    return rgb


def json2dict(filepath):
    with open(filepath) as json_file:
        data = json.load(json_file)
    return data