import streamlit as st
import pandas as pd
import numpy as np
import json, joblib

# Use st.cache_data for caching data loading functions
@st.cache_data
def load_data(fpath):
    return joblib.load(fpath)

# Use st.cache_resource for caching model loading functions
@st.cache_resource
def load_model(fpath):
    return joblib.load(fpath)

# Load file paths, the model, and the training data
with open('config/filepaths.json') as f:
    filepaths = json.load(f)

model = load_model(filepaths['models']['linear_regression'])
X_train, y_train = load_data(filepaths['data']['ml']['train'])

st.title('Home Price Prediction App')

# Creating input widgets for the features based on the training data
bathrooms = st.sidebar.slider('Bathrooms', min_value=0, max_value=int(X_train['bathrooms'].max()), step=1)
bedrooms = st.sidebar.slider('Bedrooms', min_value=0, max_value=int(X_train['bedrooms'].max()), step=1)
sqft_lot = st.sidebar.text_input('Lot Square Feet')#, min_value=int(X_train['sqft_lot'].min()), max_value=int(X_train['sqft_lot'].max()), step=1000)

# Predict button
if st.sidebar.button('Predict Price'):
    # Preparing the input data in the format the model expects
    input_data = pd.DataFrame([[bedrooms, bathrooms, sqft_lot]], columns=['bedrooms', 'bathrooms', 'sqft_lot'])
    prediction = model.predict(input_data)
    st.markdown(f'### Predicted Home Price: ${prediction[0]:,.2f}')
