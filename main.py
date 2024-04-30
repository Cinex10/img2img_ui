import PIL
import streamlit as st
import torchvision.transforms as T
from inference.edges2shoes_cyclegan.view import main as edges2shoes_cyclegan_view
from config.config import Config

def main():
    #cfg = Config(config_folder='config',config_file='config.yaml')
    initialize_state()
    
    st.set_page_config(layout="wide")
    
    add_selectbox = st.sidebar.selectbox(
        "Choose a model",
        [("Edge2shoes")]
    )
    
    image = st.sidebar.file_uploader("Upload source image", type=["png", "jpg"])

    
    st.sidebar.expander('label', expanded=True)  
    
    col1, col2 = st.columns(2,gap="small")

    with col1:
        match add_selectbox:
            case "Edge2shoes":
                edges2shoes_cyclegan_view(image)            
            case _:
                if image is not None:
                    st.image(image,width=256*2)
                else:
                    #st.info('Please upload image')
                    html_string = """<div style="text-align: center;border: thin solid rgba(255, 255, 255, 0.46);border-radius: 15px;width: 600px;height: 400px;display: flex;flex-wrap: nowrap;justify-content: center;align-items: center;">
                    Please upload image
                    </div>"""
                    st.markdown(html_string, unsafe_allow_html=True)     

    with col2:
        match st.session_state['loading_state']:
            case 'init':
                pass
            case 'loading':
                st.spinner('Wait for it...')
                    
            case 'completed':
                if (st.session_state['target'] is not None):
                    #print(st.session_state['target'].size)
                    st.image(st.session_state['target'],width=256*2)
            case _:
                html_string = '<div style="text-align: center;border: thin solid rgba(255, 255, 255, 0.46);border-radius: 15px;width: 600px;height: 400px;display: flex;flex-wrap: nowrap;justify-content: center;align-items: center;">this is an html string</div>'
                st.markdown(html_string, unsafe_allow_html=True)
    

def initialize_state():
    if 'loading_state' not in st.session_state:
        st.session_state['loading_state'] = 'init'
    if 'target' not in st.session_state:
        st.session_state['target'] = None
    if 'duaration' not in st.session_state:
        st.session_state['duaration'] = None

if __name__ == '__main__':
    main()
    
    