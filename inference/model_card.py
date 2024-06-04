import os
import torch
from . import config as cfg
from .networks import GeneratorResNet
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
        return self.module(**self.kwargs)
    
    @staticmethod
    def from_name(name):
        match name:
            case 'edges2shoes':
                return ModelCard(
                    module= GeneratorResNet,
                    pretrained_model_path=os.path.join(ROOT_DIR,cfg.MODEL_ZOO,name),
                    transform=ut.get_transform(),
                    input_shape=(3,256,256),
                    num_residual_blocks=9,
                )
            case 'colorization':
                return ModelCard(
                    module= GeneratorResNet,
                    pretrained_model_path=os.path.join(ROOT_DIR,cfg.MODEL_ZOO,'edges2shoes'),
                    transform=ut.get_transform(),
                    input_shape=(3,256,256),
                    num_residual_blocks=9,
                )
            case '1':
                pass
