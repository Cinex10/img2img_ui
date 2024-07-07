import random
import torchvision.transforms as T
from inference.pipeline import Pipeline
from inference.view import View
from config.config import Config
from streamlit_image_select import image_select
import streamlit as st

st.set_page_config(layout="centered")


choice = image_select("Choose a style", ["assets/shoe.jpeg", "assets/bag.jpeg","assets/furniture.jpeg"],return_value='index',use_container_width=True)

def initialize_state():
    if 'loading_state' not in st.session_state:
        st.session_state['loading_state'] = 'init'
    if 'target' not in st.session_state:
        st.session_state['target'] = None
    if 'duration' not in st.session_state:
        st.session_state['duration'] = None
    if 'init_draw' not in st.session_state:
        st.session_state['init_draw'] = None

initial_draw = {}

initialize_state()

# uploaded_image = None
# uploaded_image =  st.sidebar.file_uploader("Upload source image", type=["png", "jpg"])

models = {
"edges2shoes" : View('edges2shoes',can_draw=True),
 "edge2furniture" : View('edge2furniture',can_draw=True),
}

selected_model = list(Pipeline.options)[choice]

def clear():
    st.session_state['init_draw'] = None
    
def random_draw():
    predraws = models[selected_model].predraws
    if len(predraws) > 0:
        st.session_state['init_draw'] = random.choice(predraws)
    # print(initial_draw)
#selected_model = st.sidebar.selectbox(
#    "Choose a model",
#    Pipeline.options
#)


#st.sidebar.button('Generate',on_click= models[selected_model].generate)


#st.sidebar.expander('label', expanded=True)  

model = st.radio(
    "Choose a model",
    models[selected_model].available_models,
    horizontal=True,
)

col1, col2, col3 = st.columns([2,1,2],gap="small")

with col1:
    models[selected_model].render(st.session_state['init_draw'])
with col2:
    if st.button('Translate',use_container_width=True,type='primary'):
        models[selected_model].generate(model)
    st.button('Clear',use_container_width=True,on_click= clear,)
    st.button('Random',use_container_width=True,on_click= random_draw,)
    
with col3:
    # st.image('assets/bag.jpeg',width=256*2,use_column_width='always')
    match st.session_state['loading_state']:
       case 'init':
           pass
       case 'loading':
           st.spinner('Wait for it...')
               
       case 'completed':
           if (st.session_state['target'] is not None):
               #print(st.session_state['target'].size)
               st.image(st.session_state['target'],width=256)
       case _:
           html_string = '<div style="text-align: center;border: thin solid rgba(255, 255, 255, 0.46);border-radius: 15px;width: 256px;height: 256px;display: flex;flex-wrap: nowrap;justify-content: center;align-items: center;">this is an html string</div>'
           st.markdown(html_string, unsafe_allow_html=True)
 
if (st.session_state['duration'] is not None):
    st.success(f'Time elapsed {st.session_state["duration"]:.2f} sec',icon='🚀')


    