from numpy.core.records import record
import pandas as pd
import numpy as np
import cv2
import mediapipe as mp
import time
from PIL import Image
import tempfile
from bokeh.models.widgets import Div
import tensorflow as tf
from keras.preprocessing import image
from io import BytesIO
# import streamlit as st
MODEL= tf.keras.models.load_model("./nsfw_AI.h5")
CLASS_NAMES =["Normal","Semi_Nude"]

def predict(imag):
    
    img_array = tf.keras.preprocessing.image.img_to_array(imag)
    img_array = tf.expand_dims(img_array, 0)

    predictions = MODEL.predict(img_array)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence



imag = 'demo.jpg'
# img = image.load_img(img_path,color_mode='rgb', target_size=(224, 224)
img = image.load_img(imag,color_mode='rgb', target_size=(256, 256))
img_array = image.img_to_array(img)
a,b= predict(img_array)
print(a)
# print(image.shape)
