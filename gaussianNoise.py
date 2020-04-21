import glob
import random
from PIL import Image, ImageOps
from resizeimage import resizeimage
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import skimage
import cv2
from scipy.ndimage import gaussian_filter

image = plt.imread("images_processed_2/lungs/CO_13_14_1.png")

#image --> #image_out

from PIL import Image
from PIL import ImageFilter

# Open an already existing image
# Create kernel

min_blur_sharpen = -10.0
max_blur_sharpen = 10.0

filter =10#random.uniform(min_blur_sharpen, max_blur_sharpen)
if filter > 0:
    kernel = np.array([[0, -1, 0],[-1, 5,-1],[0, -1, 0]])
    image_sharp = cv2.filter2D(image, -1, kernel)
    image_sharp = cv2.filter2D(image_sharp, -1, kernel)
    image_out = (filter/10)*image_sharp + (1 - (filter/10))*image
else:
    image_out = gaussian_filter(image, sigma=round(-filter))
print(image_out)

print(filter)
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.subplot(1, 2, 2)
plt.imshow(image_out)
plt.show()
