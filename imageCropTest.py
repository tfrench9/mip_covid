import glob
import random
from PIL import Image, ImageOps
from resizeimage import resizeimage
import matplotlib.pyplot as plt
import pandas as pd
import os

for filename in glob.glob('images_processed_2/*.png'):
    image = plt.imread(filename)
    image[250:750, 250:750, 0] += 30
    plt.imshow(image)
    plt.show()
