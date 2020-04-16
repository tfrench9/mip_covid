import glob
import random
from PIL import Image, ImageOps
from resizeimage import resizeimage
import matplotlib.pyplot as plt
import pandas as pd


meta = pd.read_csv("metadata.csv")[['patientid', 'finding', 'filename']]

i = []
types = []

for row in meta.iterrows():
    pid = int(row[1][0])
    t = row[1][1]
    if not (t in types):
        types.append(t)
        i.append(0)
    filename = row[1][2]
    ti = types.index(t)

    try:

        image = Image.open("images_newset/" + filename)
        imageNew = image.resize((1000, 1250))
        if (types[ti][:2] == 'E.'):
            imageNew.save('images_processed_2/{}_{}_{}.png'.format("EC", pid, i[ti]))
        else:
            imageNew.save('images_processed_2/{}_{}_{}.png'.format(types[ti][:2], pid, i[ti]))
        i[ti] += 1

        print('Image saved: images_processed_2/{}_{}_{}.png'.format(types[ti][:2], pid, i[ti]))
        image.close()
    except:
        print("could not open: images_newset/" + filename)
