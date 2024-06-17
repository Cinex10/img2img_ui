from glob import glob
import json
import os
from typing import Any
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from .pipeline import Pipeline
from streamlit.runtime.uploaded_file_manager import UploadedFile
import torchvision.transforms as T
from PIL import Image
import pdb
from . import utils as ut

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


class View:
    def __init__(self,name,uploaded_image=None,can_draw:bool=False,**kwargs):
        self.can_draw = can_draw
        self.uploaded_image = uploaded_image
        self.pipeline = Pipeline(name)
        self.available_models = self.pipeline.available_models
        self.predraws = [ut.json2dict(x) for x in glob(os.path.join(ROOT_DIR,'predrawn_canvas',name,'*'))]
        
    def render(self,initial_draw=None, *args: Any, **kwds: Any) -> Any:
        if self.uploaded_image is not None:
            st.image(image=self.uploaded_image,width=256*1.3)
            return
        if self.can_draw:
             self.draw = st_canvas(  # Fixed fill color with some opacity
                                  stroke_width=2,
                                  stroke_color="#000",
                                  background_color="#fff",
                                  height=256,
                                  width=256,
                                  drawing_mode="freedraw",
                                  display_toolbar=True,
                                  key="full_app",  
                                  initial_drawing=initial_draw,
                                #   background_image=Image.open('bg.png')
                                  )
            #  print(self.draw.json_data)
             return
        html_string = """<div style="text-align: center;border: thin solid rgba(255, 255, 255, 0.46);border-radius: 15px;width: 512px;height: 512px;display: flex;flex-wrap: nowrap;justify-content: center;align-items: center;">Please upload image</div>"""
        st.markdown(html_string, unsafe_allow_html=True)
    
    def get_input(self):
        if self.uploaded_image is not None:
            return Image.open(self.uploaded_image)
        if self.can_draw:
            return self.draw.image_data
        raise Exception('No input')
    
    def generate(self,model='cyclegan'):
        try:
            input = self.get_input()
            st.session_state['loading_state'] = 'loading'
            duration,target = self.pipeline(input,model)
            st.session_state['target'] = target
            st.session_state['duration'] = duration
            st.session_state['loading_state'] = 'completed'
        except:
            return