import PIL
import streamlit as st
import torchvision.transforms as T
from inference.pipeline import Pipeline
from inference.view import View
from config.config import Config

st.set_page_config(layout="wide")

def initialize_state():
    if 'loading_state' not in st.session_state:
        st.session_state['loading_state'] = 'init'
    if 'target' not in st.session_state:
        st.session_state['target'] = None
    if 'duration' not in st.session_state:
        st.session_state['duration'] = None

initialize_state()

uploaded_image = st.sidebar.file_uploader("Upload source image", type=["png", "jpg"])

models = {
"colorization" : View('colorization',uploaded_image=uploaded_image),    
#  "night2day" : View('night2day',uploaded_image=uploaded_image),
#  "day2night" : View('day2night',uploaded_image=uploaded_image),
#  "label2gta" : View('label2gta',uploaded_image=uploaded_image),
#   "gta2city" : View('gta2city',uploaded_image=uploaded_image),
#  "city2gta" : View('city2gta',uploaded_image=uploaded_image),
}


selected_model = st.sidebar.selectbox(
    "Choose a model",
    Pipeline.options
)
st.sidebar.button('Generate',on_click= models[selected_model].generate)




st.sidebar.expander('label', expanded=True)  

col1, col2 = st.columns(2,gap="small")

with col1:
    models[selected_model].render()
with col2:
    
    match st.session_state['loading_state']:
        case 'init':
            pass
        case 'loading':
            st.spinner('Wait for it...')
                
        case 'completed':
            if (st.session_state['target'] is not None):
                #print(st.session_state['target'].size)
                st.image(st.session_state['target'],width=256*2,clamp=False)
        case _:
            html_string = '<div style="text-align: center;border: thin solid rgba(255, 255, 255, 0.46);border-radius: 15px;width: 600px;height: 400px;display: flex;flex-wrap: nowrap;justify-content: center;align-items: center;">this is an html string</div>'
            st.markdown(html_string, unsafe_allow_html=True)
 
if (st.session_state['duration'] is not None):
    st.success(f'Time elapsed {st.session_state["duration"]:.2f} sec',icon='🚀')


    
    