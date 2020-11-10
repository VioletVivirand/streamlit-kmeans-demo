import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
from utils import get_image, kmeans_modeling
from datetime import datetime

st.set_page_config(
    page_title="K-Means Clustering Color Quantization",
    page_icon="🧊",
    layout="wide",
    # initial_sidebar_state="expanded",
)

st.sidebar.title("Introduction")
st.sidebar.markdown("""
Submit an image with its URL, get the quantized photo back processed with K-Means algorithm 🖼

Reference: [Color Quantization using K-Means - Scikit-learn](https://scikit-learn.org/stable/auto_examples/cluster/plot_color_quantization.html#sphx-glr-auto-examples-cluster-plot-color-quantization-py)
""")
st.title('K-Means Clustering Color Quantization')

with st.beta_container():
    URL = st.text_input('Assign an image URL (Allows JPG, JPEG format only):', value='', max_chars=None, key=None, type='default')
    num_input_col1, _, num_input_col2 = st.beta_columns([5,1,5])
    n_colors = num_input_col1.select_slider('Number of colors', [4,8,16,32,64,128,256], value=64)
    n_samples = num_input_col2.select_slider('Number of samples', [10,100,1000,10000], value=1000)
    url_submit_btn = st.button('Submit')

if url_submit_btn:
    with st.beta_container():
        photo_col1, photo_col2, photo_col3 = st.beta_columns([4,4,4])

    if URL:
        photo_doc = get_image(URL)
        if isinstance(photo_doc['photo'], np.ndarray):
            photo_col1.image(photo_doc['photo'], caption=photo_doc['caption'], use_column_width=True)

            photo2_doc, photo3_doc = kmeans_modeling(
                photo_doc['photo'],
                n_colors=n_colors,
                n_samples=n_samples,
            )
            photo_col2.image(photo2_doc['photo'], caption=photo2_doc['caption'], use_column_width=True)
            photo_col3.image(photo3_doc['photo'], caption=photo3_doc['caption'], use_column_width=True)
        else:
            pass
