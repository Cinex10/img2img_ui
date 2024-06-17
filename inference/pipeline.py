import gc
from glob import glob
import os
import pdb
import torch
from typing import Any
from .model_card import ModelCard
from .networks import GeneratorResNet
from . import utils as ut
from PIL import Image
from . import config as cfg


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

class Pipeline:
    options = set()
    def __init__(self,domain) -> None:
        Pipeline.options.add(domain)
        self.domain : str = domain
        self.model_cards :dict[str,ModelCard] = ModelCard.from_name(domain)
        self.pretrained_models_path=os.path.join(ROOT_DIR,cfg.MODEL_ZOO,domain)
        self.available_models = [x.split('/')[-1].split('.')[0] for x in glob(self.pretrained_models_path+'/*.pth')]
        
    def load(self,model):
        model_path = glob(self.pretrained_models_path+f'/{model}.pth')[0]
        # pdb.set_trace()
        self.model : torch.nn.Module = self.model_cards[model].get_model()
        self.model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
        self.model.eval()
    
    def __call__(self,source,model,*args: Any, **kwds: Any) -> Any:
        x1 = ut.get_input_image(source)
        x = self.preprocess(x1,model)
        self.load(model)
        duration,y = self.forward(x)
        print(duration)
        if (model == 'pix2pix') & (self.domain == 'colorization'):
            print('grayscale')
            y = ut.lab2rgb(x,y)
        else:
            y = self.postprocessing(y)
        del self.model
        gc.collect()
        return duration,y
        
        
    
    def preprocess(self,source,model):
        transform = self.model_cards[model].transfrom
        source = transform(source)
        # pdb.set_trace()
        return source.unsqueeze(0)

    def postprocessing(self,source):
        source = ut.tensor2img(source)
        return Image.fromarray(source)
    
    @ut.get_runtime
    def forward(self,source):
        return self.model(source)

