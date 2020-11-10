import requests
from PIL import Image
from io import BytesIO
import numpy as np
import streamlit as st

def get_image(url):
    try:
        r = requests.get(url)
        img = Image.open(BytesIO(r.content))
        photo = np.array(img)
        st.write('Shape of photo: {}'.format(photo.shape))
        _, _, d = tuple(photo.shape)
        assert d == 3

        # (Additional step) Calculate the number of colors in the photo
        w, h, d = original_shape = tuple(photo.shape)
        image_array = np.reshape(photo, (w * h, d))
        photo_n_colors = np.unique(image_array, axis=0).shape[0]

        return {           
            'photo': photo,
            'caption': 'Original image ({photo_n_colors} colors)'.format(photo_n_colors=photo_n_colors),
        }
    except Exception as e:
        st.error('Invalid image.')
        return {           
            'photo': '',
            'caption': '',
        }
