import streamlit as st
from streamlit_drawable_canvas import st_canvas
from .predictor import Edges2shoesPredictor
from streamlit.runtime.uploaded_file_manager import UploadedFile
import torchvision.transforms as T
from PIL import Image
import pdb


    

edges2shoes_predictor = Edges2shoesPredictor()

def main(uploaded_image:UploadedFile):
    canvas_result = None
    if uploaded_image is not None:
        st.image(image=uploaded_image,width=256*2)
    else:
        canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=2,
        stroke_color="#000",
        background_color="#fff",
        height=256*2,
        width=256*2,
        drawing_mode="freedraw",
        display_toolbar=True,
        key="full_app",
        )
    
    
    def generate_image():
        input = None
        if uploaded_image is not None:
            input = Image.open(uploaded_image)
        else:
            input = canvas_result.image_data
        st.session_state['loading_state'] = 'loading'
        #breakpoint()
        duration,target = edges2shoes_predictor.translate(input)
        st.session_state['target'] = target
        st.session_state['duration'] = duration
        #print(target)
        st.session_state['loading_state'] = 'completed'
    
    st.sidebar.button('Generate',on_click= generate_image)
 
    

