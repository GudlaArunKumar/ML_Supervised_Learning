"""
Python program file which takes input from the user 
and display predictions in web application using Streamlit

Author - Arun Kumar Gudla 
"""


# importing all basic libraries
import streamlit as st
import pandas as pd
import numpy as np
import pickle


# loading the trained model
file1 = open('rf_pipe.pkl', 'rb')
rf = pickle.load(file1) 
file1.close()

# reading the trained dataset 
data = pd.read_csv('trainned_data.csv')


# Defining frontend elemnts using streamlit 
# defining all features which are present in train data
st.title('Laptop Price Predictor')

# company of laptop
company = st.selectbox('Brand', data['Company'].unique())

# type of laptop
type = st.selectbox('Type', data['TypeName'].unique())

# Ram 
ram = st.selectbox('Ram (in GB)', [2, 4 ,6, 8, 12, 16, 24, 32, 64])

# Operating system
os = st.selectbox('OS', data['OpSys'].unique())

# Weight of laptop 
weight = st.number_input("Weight of the laptop (in Kg)")

# Touch screen option
touchscreen = st.selectbox("Touch Screen", ['Yes', 'No'])

# IPS 
ips = st.selectbox("IPS", ['Yes', 'No'])

# screen size (inches)
screen_size = st.number_input("Screen size (in inches")

#resolution of the latop
resolution = st.selectbox("Resolution", ['1920x1080', '1366x768', 
'1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'])

# cpu
cpu = st.selectbox('CPU', data['CPU_name'].unique())

# hdd
hdd = st.selectbox('HDD(in GB)', [0, 128, 256, 512, 1024, 2048])

# ssd
ssd = st.selectbox('SSD(in GB)', [0, 8, 128, 256, 512, 1024])

# GPU
gpu = st.selectbox('GPU(in GB)', data['Gpu_brand'].unique())

# display an background image for effective visual
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("https://res.cloudinary.com/dhhasit7z/image/upload/v1674330496/laptop_image_1_ibl659.webp");
        background-attachment: fixed;
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# prediction on button press
if st.button('Predict Price'):

    # making touch screen feature as integer
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    # making IPS panel feature as integer
    if ips == 'Yes':
        ips = 1 
    else:
        ips = 0 
    
    # calculating PPI which is required feature for the dataset 
    X_resolution = int(resolution.split('x')[0])
    Y_resolution = int(resolution.split('x')[1])

    ppi = ((X_resolution)**2 + (Y_resolution)**2)**0.5 / (screen_size)

    # grouping all input data
    query = np.array([company, type, ram, os, weight, touchscreen, ips, ppi, cpu, 
    hdd, ssd, gpu])

    query = query.reshape(1,12)

    # since the output prediction will be in log, converting back to INR
    prediction = int(np.exp(rf.predict(query)[0]))


    # displaying output in Text box
    modelresponse = "Predicted price for this laptop could be between " + str(prediction-1000) + "₹" + " to " + str(prediction+1000) + "₹"
    st.text_area(label ="", value=modelresponse, height = 25)

    

