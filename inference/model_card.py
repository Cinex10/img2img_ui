import os
import torch
from .networks import define_G, define_G_pixHD
from . import utils as ut
from PIL import Image

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
                ),
                    'pix2pixHD' : ModelCard(
                    module= define_G_pixHD(3, 3, 64, 'global', 4, 9,1, 3, 'instance'),
                    transform=ut.get_pixHD_trans(resize_or_crop='scale_width',n_downsample_global=4,fineSize=512,loadSize=1024,n_local_enhancers=1,netG='global',params=ut.get_params('scale_width',1024,512,(256,256)), method=Image.NEAREST, normalize=False),
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
