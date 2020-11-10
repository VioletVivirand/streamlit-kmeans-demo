import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle
from time import time

def recreate_image(codebook, labels, w, h):
    """Recreate the (compressed) image from the code book & labels"""
    d = codebook.shape[1]
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    return image

def kmeans_modeling(photo, n_colors=64, n_samples=1000):
    progress_bar = st.progress(0)

    # (Additional step) Calculate the number of colors in the photo
    w, h, d = original_shape = tuple(photo.shape)
    image_array = np.reshape(photo, (w * h, d))
    photo_n_colors = np.unique(image_array, axis=0).shape[0]

    # Convert to floats instead of the default 8 bits integer coding. Dividing by
    # 255 is important so that plt.imshow behaves works well on float data (need to
    # be in the range [0-1])
    photo = np.array(photo, dtype=np.float64) / 255

    # Transform to a 2D numpy array.
    image_array = np.reshape(photo, (w * h, d))

    st.write("Fitting model on a small sub-sample of the data")
    t0 = time()
    image_array_sample = shuffle(image_array, random_state=0)[:n_samples]
    kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)
    st.write("done in %0.3fs." % (time() - t0))

    # Get labels for all points
    st.write("Predicting color indices on the full image (k-means)")
    t0 = time()
    labels = kmeans.predict(image_array)
    st.write("done in %0.3fs." % (time() - t0))
    progress_bar.progress(25)
    kmeans_image = recreate_image(kmeans.cluster_centers_, labels, w, h)
    progress_bar.progress(50)

    codebook_random = shuffle(image_array, random_state=0)[:n_colors]
    st.write("Predicting color indices on the full image (random)")
    t0 = time()
    labels_random = pairwise_distances_argmin(codebook_random,
                                            image_array,
                                            axis=0)
    st.write("done in %0.3fs." % (time() - t0))
    progress_bar.progress(75)
    kmeans_random_image = recreate_image(codebook_random, labels_random, w, h)
    progress_bar.progress(100)

    return ({
            'photo': kmeans_image,
            'caption': 'Quantized image ({n_colors} colors, K-Means)'.format(n_colors=n_colors),
        }, {
            'photo': kmeans_random_image,
            'caption': 'Quantized image ({n_colors} colors, Random)'.format(n_colors=n_colors)
        }
    )
