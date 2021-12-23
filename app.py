# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 16:47:59 2021

@author: ishan
"""
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import model_from_json
import cv2
from cv2 import *
import streamlit as st
from PIL import Image,ImageOps
import numpy as np

global json_file
global loaded_model
global loaded_model_json
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
 # load weights into new model
loaded_model.load_weights("model.h5")

def teachable_machine_classification(img):
    
# Load the model
    



# Create the array of the right shape to feed into the keras model
    
    image = img
#image sizing
    size = (100,100)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)



#turn the image into a numpy array
    image_array = np.asarray(image)

    image=image_array.reshape(1,10000)



# Load the image into the array
    
    prediction=loaded_model.predict(image)
    return prediction[0][0]



st.title("Boy or Girl?")

uploaded_file = st.file_uploader("Upload an Image...", type=["jpg","png"])
if uploaded_file is not None:
    image=Image.open(uploaded_file)
    image1=image
    image = ImageOps.grayscale(image)
    label=teachable_machine_classification(image)
   
    if label<0.5:
        new_title = '<p style="font-family:sans-serif; color:Pink; font-size: 42px;">It is a Girl</p>' 
        st.markdown(new_title, unsafe_allow_html=True)
    else:
        new_title = '<p style="font-family:sans-serif; color:Blue; font-size: 42px;">It is a Boy</p>' 
        st.markdown(new_title, unsafe_allow_html=True)
    
    st.image(image1)
    
   
    
