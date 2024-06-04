import torchvision.transforms as T
from inference.pipeline import Pipeline
from inference.view import View
from config.config import Config
from streamlit_image_select import image_select
import streamlit as st

st.set_page_config(layout="wide")

_,c,_ = st.columns([1,3,1])
with c:
    choice = image_select("Choose a style", ["assets/bag.jpeg", "assets/cat.jpeg", "assets/shoe.jpeg","assets/furniture.jpeg"],return_value='index',use_container_width=True)

def initialize_state():
    if 'loading_state' not in st.session_state:
        st.session_state['loading_state'] = 'init'
    if 'target' not in st.session_state:
        st.session_state['target'] = None
    if 'duration' not in st.session_state:
        st.session_state['duration'] = None

initialize_state()

uploaded_image = None

models = {
"edges2shoes" : View('edges2shoes',can_draw=True,uploaded_image=uploaded_image),
}

selected_model = Pipeline.options[choice]

#selected_model = st.sidebar.selectbox(
#    "Choose a model",
#    Pipeline.options
#)


#st.sidebar.button('Generate',on_click= models[selected_model].generate)


#st.sidebar.expander('label', expanded=True)  

col1, col2, col3 = st.columns(3,gap="large")

with col1:
    models[selected_model].render()
with col2:
    
    _, col4, _ = st.columns(3,gap="large")
    with col4:
        st.button('Translate',use_container_width=True)
with col3:
    st.image('assets/bag.jpeg',width=256*2,use_column_width='always')
    #match st.session_state['loading_state']:
    #    case 'init':
    #        pass
    #    case 'loading':
    #        st.spinner('Wait for it...')
    #            
    #    case 'completed':
    #        if (st.session_state['target'] is not None):
    #            #print(st.session_state['target'].size)
    #            st.image(st.session_state['target'],width=256*2)
    #    case _:
    #        html_string = '<div style="text-align: center;border: thin solid rgba(255, 255, 255, 0.46);border-radius: 15px;width: 600px;height: 400px;display: flex;flex-wrap: nowrap;justify-content: center;align-items: center;">this is an html string</div>'
    #        st.markdown(html_string, unsafe_allow_html=True)
 
if (st.session_state['duration'] is not None):
    st.success(f'Time elapsed {st.session_state["duration"]:.2f} sec',icon='🚀')


    