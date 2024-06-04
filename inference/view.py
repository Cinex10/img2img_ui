from typing import Any
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from .pipeline import Pipeline
from streamlit.runtime.uploaded_file_manager import UploadedFile
import torchvision.transforms as T
from PIL import Image
import pdb


class View:
    def __init__(self,name,uploaded_image,can_draw:bool=False,**kwargs):
        self.can_draw = can_draw
        self.uploaded_image = uploaded_image
        self.pipeline = Pipeline(name)
    
    def render(self, *args: Any, **kwds: Any) -> Any:
        if self.uploaded_image is not None:
            st.image(image=self.uploaded_image,width=256*2)
            return
        if self.can_draw:
             self.draw = st_canvas(fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
                                  stroke_width=2,
                                  stroke_color="#000",
                                  background_color="#fff",
                                  height=256*2,
                                  width=256*2,
                                  drawing_mode="freedraw",
                                  display_toolbar=True,
                                  key="full_app",
                                  )
             return
        html_string = """<div style="text-align: center;border: thin solid rgba(255, 255, 255, 0.46);border-radius: 15px;width: 600px;height: 400px;display: flex;flex-wrap: nowrap;justify-content: center;align-items: center;">Please upload image</div>"""
        st.markdown(html_string, unsafe_allow_html=True)
    
    def get_input(self):
        if self.uploaded_image is not None:
            return Image.open(self.uploaded_image)
        if self.draw is not None:
            return self.draw.image_data
        raise Exception('No input')
    
    def generate(self):
        input = self.get_input()
        st.session_state['loading_state'] = 'loading'
        duration,target = self.pipeline(input)
        st.session_state['target'] = target
        st.session_state['duration'] = duration
        st.session_state['loading_state'] = 'completed'
