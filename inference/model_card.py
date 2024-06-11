import os
import torch
from . import config as cfg
from .networks import GeneratorResNet,define_G
from . import utils as ut

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
class ModelCard:
    def __init__(self,
                 module:torch.nn.Module,
                 pretrained_model_path : str,
                 transform,
                 **kwargs,
                 ) -> None:
        self.module = module
        self.path = pretrained_model_path
        self.transfrom = transform
        self.kwargs = kwargs
    
    def get_model(self):
        # return self.module(**self.kwargs)
        return self.module
    
    @staticmethod
    def from_name(name):
        match name:
            case 'edges2shoes':
                return ModelCard(
                    module= define_G(3,3,64,'resnet_9blocks','instance',False,'normal',0.02),
                    pretrained_model_path=os.path.join(ROOT_DIR,cfg.MODEL_ZOO,name),
                    transform=ut.get_transform(),
                    input_shape=(3,256,256),
                    num_residual_blocks=9,
                )
            case 'colorization':
                return ModelCard(
                    module= define_G(1,2,64,'unet_256','batch',True,'normal',0.02),
                    pretrained_model_path=os.path.join(ROOT_DIR,cfg.MODEL_ZOO,name),
                    transform=ut.get_transform(colorization=True),
                    input_shape=(3,256,256),
                    num_residual_blocks=9,
                )
            case 'night2day':
                return ModelCard(
                    module= define_G(3,3,64,'resnet_9blocks','instance',False,'normal',0.02),
                    pretrained_model_path=os.path.join(ROOT_DIR,cfg.MODEL_ZOO,name),
                    transform=ut.get_transform(),
                )
            case 'day2night':
                return ModelCard(
                    module= define_G(3,3,64,'resnet_9blocks','instance',False,'normal',0.02),
                    pretrained_model_path=os.path.join(ROOT_DIR,cfg.MODEL_ZOO,name),
                    transform=ut.get_transform(),
                )
            case 'label2gta':
                return ModelCard(
                    module= define_G(3,3,64,'unet_256','batch',False,'normal',0.02),
                    pretrained_model_path=os.path.join(ROOT_DIR,cfg.MODEL_ZOO,name),
                    transform=ut.get_transform(),
                )
            case 'gta2city':
                return ModelCard(
                    module= define_G(3,3,64,'resnet_9blocks','instance',False,'normal',0.02),
                    pretrained_model_path=os.path.join(ROOT_DIR,cfg.MODEL_ZOO,name),
                    transform=ut.get_transform(),
                )
            case 'city2gta':
                return ModelCard(
                    module= define_G(3,3,64,'resnet_9blocks','instance',False,'normal',0.02),
                    pretrained_model_path=os.path.join(ROOT_DIR,cfg.MODEL_ZOO,name),
                    transform=ut.get_transform(),
                )
            case '1':
                pass
