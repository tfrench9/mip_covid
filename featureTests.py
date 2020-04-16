from skimage.color import rgb2hsv, rgb2lab, rgb2gray
from scipy.fftpack import dct
from skimage.feature import canny, greycomatrix, greycoprops
from skimage.measure import shannon_entropy
import cv2
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import scipy.misc
import matplotlib.pyplot as plt
import os
import random


images = os.listdir('images_processed_2')
random.shuffle(images)
images = images[:40]
data = np.zeros([len(images), 1250, 1000])
labels = np.zeros(len(images))
for i in range(len(images)):
    arr = plt.imread('images_processed_2/{}'.format(images[i]))
    gray = rgb2gray(arr)
    data[i, :, :] = gray
    if 'CO' in images[i]:
        labels[i] = 0
    else:
        labels[i] = 1

covid = data[labels == 0, :, :]
for i in range(5):
    plt.subplot(2, 5, i + 1)
    plt.imshow(covid[i])
    plt.subplot(2, 5, i + 6)
    dct2 = dct(dct(covid[i], axis = 0), axis = 1)
    plt.imshow(np.log(np.absolute(dct2)))

plt.show()
