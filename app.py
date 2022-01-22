import numpy as np
from PIL import Image,ImageOps, ImageFilter
import tensorflow as tf
import streamlit as st
from itertools import cycle
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="NSFW-AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded")
# ---------------------------------------------------------------------
st.title('NSFW-AI')
MODEL= tf.keras.models.load_model("./nsfw_AI.h5")
CLASS_NAMES =["Normal","Semi_Nude"]
DEMO_IMAGE= 'demo.jpg'
# ---------------------------------------------------------------------
def predict(imag):
    size = (256,256)    
    image = ImageOps.fit(imag, size, Image.ANTIALIAS)
    image = np.asarray(image)
    img_reshape = image[np.newaxis,...]
    predictions = MODEL.predict(img_reshape)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence
# ---------------------------------------------------------------------
app_mode= st.selectbox('Choose the App Mode',
                               ['About App','Run Tests','Show Code'])

st.markdown ('---------' )
# ---------------------------------------------------------------------

if app_mode== 'About App':
    text= '''App Model Made with **Tensorflow** & **Keras** using **Convolutional Neural Networks. 
                The Model is trained on 2500+ Images which include 1200+ Normal Images & 1000+ Semi Nude Images of Mens. There are many Factors like **Races**,**Clothes **Image Lighting** etc. but still model is 80-90% Accurate The Data is Scrapped from google.More the Data, More the Accuracy. **Dataset is not available on Kaggle** '''
    st.markdown(text)
    
    st.write("Check out my GitHub [link](https://github.com/shashankanand13monu)")
    st.write("My Portfolio [link](https://shashankanand13mon.wixsite.com/portfolio)")
    
elif app_mode == 'Run Tests':
# Try & Except Fix
    try :
        
        img_file_buffer = st.file_uploader("Upload an Image", type=["jpg","jpeg", "png"])
        if img_file_buffer is not None:
            imag = Image.open(img_file_buffer)
        else:
            demo_image= DEMO_IMAGE
            imag= Image.open(demo_image)
            
        st.subheader('Predictions')
            
        kpi1, kpi2, kpi3 = st.columns(3)
        with kpi1:
            original_title = '<p style="text-align: center; font-size: 20px;"><strong>Label</strong></p>'
            st.markdown(original_title, unsafe_allow_html=True)
            kpi1_text = st.markdown ("0")
        with kpi2:
            original_title = '<p style="text-align: center; font-size: 20px;"><strong>Confidence</strong></p>'
            st.markdown(original_title, unsafe_allow_html=True)
            kpi2_text = st.markdown ("0")    # imag = Image.open(imag)
        st.subheader('Input Image')
        category,confidence = predict(imag)
        
        col1, col2,col3,col4 = st.columns(4)
        original = imag
        
        original_title = '<p style="font-family:Arial Black; font-size: 40px;">Original</p>'
        col1.markdown(original_title, unsafe_allow_html=True)
        col1.image(original, use_column_width=True,width=256)
    # ---------------------------------------------------------------------
        action1= '''<style >
        p.detail { font-weight:bold;font-family:Arial Black;font-size:40px }
        span.name { color:#06FD01;font-weight:bold;font-family:Tahoma;font-size:40px }
        </style>  <p class="detail">Action: <span class="name">None</span> </p>'''
        action2= '''<style >
        p.detail { font-weight:bold;font-family:Arial Black;font-size:40px }
        span.name { color:#FD0101;font-weight:bold;font-family:Tahoma;font-size:40px }
        </style>  <p class="detail">Action: <span class="name">Blured</span> </p>'''
    # ---------------------------------------------------------------------
        
        if category=='Normal':
            blurImage= imag
            original_title = '<p style="font-family:Arial Black; color:#FD0177; font-size: 40px;">Pass</p>'
            col2.markdown(action1, unsafe_allow_html=True)
        else:   
            blurImage = original.filter(ImageFilter.GaussianBlur(5))
            original_title = '<p style="font-family:Arial Black; color:#FD0177; font-size: 40px;">Blurred</p>'
        
            col2.markdown(action2, unsafe_allow_html=True)
            
        col2.image(blurImage, use_column_width=True,width=256) 
        kpi1_text.write(f"<h1 style='text-align: center; color:red; '>{category}</h1>", unsafe_allow_html=True)
        kpi2_text.write(f"<h1 style='text-align: center; color:red; '>{int(confidence)}%</h1>", unsafe_allow_html=True)    
        st.subheader("Demo Images")
        
        filteredImages = ['00000215.jpg','00001670.jpg','underwearmens29.jpeg','hell.jpg','00000062.jpg','00000012.jpg'] # your images here
        # caption = [] # your caption here
        cols = cycle(st.columns(5)) # st.columns here since it is out of beta at the time I'm writing this
        for idx, filteredImage in enumerate(filteredImages):
            next(cols).image(filteredImage, width=150)
    except:
        st.subheader("Oops, Image not compatible")
        st.subheader("Dmo Images : ")
        filteredImages = ['00000215.jpg','00001670.jpg','underwearmens29.jpeg','hell.jpg','00000062.jpg','00000012.jpg'] # your images here
        # caption = [] # your caption here
        cols = cycle(st.columns(5)) # st.columns here since it is out of beta at the time I'm writing this
        for idx, filteredImage in enumerate(filteredImages):
            next(cols).image(filteredImage, width=150)
# ---------------------------------------------------------------------
elif app_mode== 'Show Code':
    
    agree = st.checkbox('Show Full Training Code')
    if agree:
        st.subheader('Traing GitHub Code') 
        gist= '''<script src="https://gist.github.com/shashankanand13monu/c52978af359ea32a6cdc2a9e83b447c1.js"></script>'''
        st.components.v1.html(gist,width=1024,height=1000,scrolling=True)
    else:
        
        
        code= '''<script src="https://gist.github.com/shashankanand13monu/98d05f79957d8460cada91aea1e19015.js"></script>'''

        st.components.v1.html(code,width=1024,height=1000,scrolling=True)
