import glob
import random
from PIL import Image, ImageOps
from resizeimage import resizeimage
import matplotlib.pyplot as plt
import pandas as pd
import os

meta = pd.read_csv("Data_Entry_2017.csv")[['Patient ID', 'Finding Labels', 'Image Index', 'View Position']]

i = 13
types = []

for row in meta.iterrows():
    pid = int(row[1][0])
    t = row[1][1]
    if("Pneumo" in t):
        filename = row[1][2]
        view = row[1][3]

        if(view == "PA"):
            try:
                if not os.path.exists('images_processed_2/{}_{}_{}.png'.format("Pn", pid, i)):
                    image = Image.open("images_newset_2/" + filename)
                    imageNew = image.resize((1000, 1250))
                    imageNew.save('images_processed_2/{}_{}_{}.png'.format("Pn", pid, i))
                    i += 1
                    print('images_processed_2/{}_{}_{}.png'.format("Pn", pid, i))
                    image.close()
                else:
                    print("Already Here")
            except:
                print("could not open: images_newset/" + filename)
