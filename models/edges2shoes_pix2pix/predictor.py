import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import logging as log
from datetime import datetime
import torch
from .networks import define_G
from collections import OrderedDict
from PIL import Image
import torchvision.transforms as transforms 
import torch
import numpy as np
from PIL import Image
import os


def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)



class Edges2shoesPredictor:
 
    def __init__(self,channels,img_height,img_width,n_residual_blocks):
        self.output_dir = 'results'
        self.img_height = img_height
        self.img_width = img_width
        self.channels = 3
        input_shape = (channels, img_height, img_width)
        # Initialize generator 
        path = os.path.join('models/edges2shoes_pix2pix/saved_model','latest_net_G.pth')
        model_dict = torch.load(path)
        new_dict = OrderedDict()
        for k, v in model_dict.items():
            # load_state_dict expects keys with prefix 'module.'
            #print(k)
            new_dict[k] = v
        self.generator_model = define_G(input_nc=3,output_nc=3,ngf=64,netG="unet_256",
                            norm="batch",use_dropout=False,init_gain=0.02,gpu_ids=[])
        self.generator_model.load_state_dict(new_dict)
        log.info(f'Model loaded succesfully from {path}')
        self.resize = transforms.Resize((256,256),transforms.InterpolationMode.BICUBIC)
        self.to_pil = transforms.ToPILImage(mode='RGB')
        self.transforms_ = transforms.Compose([
            transforms.Resize((286,286),transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.RandomCrop(256),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            #transforms.PILToTensor(),
        ])
    
    def predict(self,source):
        source = Image.fromarray(np.array(source)[:, ::-1, :]).convert('RGB')
        input = self.transforms_(source)
        input = input.unsqueeze(0)
        self.generator_model.eval()
        before = datetime.now()
        target = self.generator_model(input.float())
        after = datetime.now()
        diffrence = after - before
        log.info(f'Image translating in {diffrence.total_seconds()} seconds')
        #target = target.squeeze(0)
        #target = torch.flip(target,[2])
        target = tensor2im(target)
        target = Image.fromarray(target)
        #input = input.squeeze(0)
        
        self.save_image_pair(source,target)
        return target
    
    def save_image_pair(self,source,target):   
        print(type(source),type(target))
        name = datetime.now().strftime("%d_%m_%Y_%H_%M_%S.png")
        path = os.path.join(self.output_dir,name)
        img2 = Image.new("RGB", (256*2, 256)) 
        img2.paste(source,(0,0))
        img2.paste(target,(256,0))
        img2.save(path)
