from glob import glob
import pdb
import torch
from typing import Any
from .model_card import ModelCard
from .networks import GeneratorResNet
from . import utils as ut
from PIL import Image


class Pipeline:
    options = []
    def __init__(self,name) -> None:
        Pipeline.options.append(name)
        self.name : str = name
        self.model_card : ModelCard = ModelCard.from_name(name)
        self.model : torch.nn.Module = self.model_card.get_model()
        self.load()
        
    def load(self):
        model_path = glob(self.model_card.path+'/*.pth')[0]
        self.model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
        self.model.eval()
    
    def __call__(self,source, *args: Any, **kwds: Any) -> Any:
        x1 = ut.get_input_image(source)
        x = self.preprocess(x1)
        duration,y = self.forward(x)
        print(duration)
        if self.name == 'colorization':
            y = ut.lab2rgb(x,y)
        else:
            y = self.postprocessing(y)
        return duration,y
        
        
    
    def preprocess(self,source):
        transform = self.model_card.transfrom
        source = transform(source)
        pdb.set_trace()
        return source.unsqueeze(0)

    def postprocessing(self,source):
        source = ut.tensor2img(source)
        return Image.fromarray(source)
    
    @ut.get_runtime
    def forward(self,source):
        return self.model(source)

