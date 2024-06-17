import os
import torch
from .networks import define_G
from . import utils as ut

class ModelCard:
    def __init__(self,
                 module:torch.nn.Module,
                 transform,
                 **kwargs,
                 ) -> None:
        self.module = module
        self.transfrom = transform
        self.kwargs = kwargs
    
    def get_model(self):
        # return self.module(**self.kwargs)
        return self.module
    
    @staticmethod
    def from_name(name):
        match name:
            case 'edges2shoes':
                return {
                    'cyclegan' : ModelCard(
                    module= define_G(3,3,64,'resnet_9blocks','instance',False,'normal',0.02),
                    transform=ut.get_transform(),
                )
                }
            case 'colorization':
                return {
                    'pix2pix' : ModelCard(
                    module= define_G(1,2,64,'unet_256','batch',True,'normal',0.02),
                    transform=ut.get_transform(colorization=True),
                ),
                    'cyclegan' : ModelCard(
                    module= define_G(3,3,64,'resnet_9blocks','instance',False,'normal',0.02),
                    transform=ut.get_transform(),
                )
                }
            case 'night2day':
                return {
                    'cyclegan' : ModelCard(
                    module= define_G(3,3,64,'resnet_9blocks','instance',False,'normal',0.02),
                    transform=ut.get_transform(),
                )
                }
            case 'day2night':
                return {
                    'cyclegan' : ModelCard(
                    module= define_G(3,3,64,'resnet_9blocks','instance',False,'normal',0.02),
                    transform=ut.get_transform(),
                )
                }
            case 'label2gta':
                return {
                    'pix2pix' : ModelCard(
                    module= define_G(3,3,64,'unet_256','batch',False,'normal',0.02),
                    transform=ut.get_transform(),
                )
                }
            case 'edge2furniture':
                return {
                    'pix2pix' : ModelCard(
                    module= define_G(3,3,64,'unet_256','batch',True,'normal',0.02),
                    transform=ut.get_transform(),
                )
                }
            case 'gta2city':
                return {'cyclegan':ModelCard(
                    module= define_G(3,3,64,'resnet_9blocks','instance',False,'normal',0.02),
                    transform=ut.get_transform(),
                )}
            case 'city2gta':
                return {'cyclegan':ModelCard(
                    module= define_G(3,3,64,'resnet_9blocks','instance',False,'normal',0.02),
                    transform=ut.get_transform(),
                )}
