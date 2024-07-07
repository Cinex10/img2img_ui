
import functools
from datetime import datetime
import json
import os
import pdb
import random
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

def normalize():    
    return transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size        
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img
    return img.resize((w, h), method)

def __scale_width(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    if (ow == target_width):
        return img    
    w = target_width
    h = int(target_width * oh / ow)    
    return img.resize((w, h), method)

def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):        
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img

def get_pixHD_trans(resize_or_crop,n_downsample_global,n_local_enhancers,loadSize,fineSize, params,netG, method=Image.BICUBIC, normalize=True):
    transform_list = []
    if 'resize' in resize_or_crop:
        osize = [loadSize, loadSize]
        transform_list.append(transforms.Scale(osize, method))   
    elif 'scale_width' in resize_or_crop:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, loadSize, method)))
        
    if 'crop' in resize_or_crop:
        transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], fineSize)))

    if resize_or_crop == 'none':
        base = float(2 ** n_downsample_global)
        if netG == 'local':
            base *= (2 ** n_local_enhancers)
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base, method)))

    transform_list += [transforms.ToTensor()]

    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    transform_list += [transforms.Lambda(lambda x: x * 255)]
    return transforms.Compose(transform_list)

def get_params(resize_or_crop,loadSize,fineSize, size):
    w, h = size
    new_h = h
    new_w = w
    if resize_or_crop == 'resize_and_crop':
        new_h = new_w = loadSize            
    elif resize_or_crop == 'scale_width_and_crop':
        new_w = loadSize
        new_h = loadSize * h // w

    x = random.randint(0, np.maximum(0, new_w - fineSize))
    y = random.randint(0, np.maximum(0, new_h - fineSize))
    
    flip = random.random() > 0.5
    return {'crop_pos': (x, y), 'flip': flip}


def json2dict(filepath):
    with open(filepath) as json_file:
        data = json.load(json_file)
    return data