import os
import PIL
import PIL.Image
import torch
import torchvision.transforms as transforms
from PIL import Image
import logging as log
from ..networks import GeneratorResNet
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import logging as log
from inference.networks import GeneratorResNet
from inference import utils
import numpy as np



class Edges2shoesPredictor:
 
    def __init__(self,channels=3,img_height=256,img_width=256,n_residual_blocks=9):
        self.output_dir = 'results'
        self.img_height = img_height
        self.img_width = img_width
        self.channels = 3
        input_shape = (channels, img_height, img_width)
        # Initialize generator 
        self.G_AB = GeneratorResNet(input_shape, n_residual_blocks)
        path = os.path.join('inference/edges2shoes_cyclegan/saved_model','G_AB.pth')
        self.G_AB.load_state_dict(torch.load(path,map_location=torch.device('cpu')))
        self.G_AB.eval()
        log.info(f'Model loaded succesfully from {path}')
        self.transforms_ = transforms.Compose([
        transforms.Resize((256, 256), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])  
        
        
    
    def translate(self,source,to_pil=True):
        input_image = None
        if isinstance(source, str):
            input_image = Image.open(source).convert('RGB')
        if isinstance(source, np.ndarray):
            input_image = Image.fromarray(source).convert('RGB')
        if  isinstance(source,PIL.PngImagePlugin.PngImageFile):
            input_image = source
        else:
            raise Exception('Unsupported type')
        input_tensor = self.transforms_(input_image)
        input_tensor = input_tensor.unsqueeze(0)    
        duration,target_tensor = self.predict(input_tensor)
        log.info(f'Image translating in {duration} seconds')
        target_image = utils.tensor2img(target_tensor)
        if to_pil:
            target_image = Image.fromarray(target_image)
        return duration,target_image
    
    @utils.get_runtime
    def predict(self,source):
        return self.G_AB(source)
    
    def __del__():
        print('deleted          ###                 ### """"""""""""""""""""""""""""""""""oeiuhrkjhrktlrekzhtkjzehrtlkzejtlkzejtlzerktlkertj')