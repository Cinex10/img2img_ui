import streamlit as st
from streamlit_drawable_canvas import st_canvas
from .predictor import Edges2shoesPredictor
from streamlit.runtime.uploaded_file_manager import UploadedFile
import torchvision.transforms as T
from PIL import Image


    

edges2shoes_pix2pix_predictor = Edges2shoesPredictor(channels=3,img_height=256,img_width=256,n_residual_blocks = 9)

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
        target = edges2shoes_pix2pix_predictor.predict(input)
        #transform = T.ToPILImage(mode='RGB')
        st.session_state['target'] = target
        #print(target)
        st.session_state['loading_state'] = 'completed'
    
    st.sidebar.button('Generate',on_click= generate_image)
 
    

