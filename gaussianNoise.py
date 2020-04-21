import glob
import random
from PIL import Image, ImageOps
from resizeimage import resizeimage
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import skimage

image = plt.imread("images_processed_2/lungs/CO_13_14_1.png")

#image --> #image_out

strength = 10 # from 0 to 30
noise = skimage.color.grey2rgb(np.random.normal(0, strength, image.shape[:2]))
image_out = (255*image + noise).astype('uint8')/255

plt.subplot(1, 2, 1)
plt.imshow(image)
plt.subplot(1, 2, 2)
plt.imshow(image_out)
plt.show()
