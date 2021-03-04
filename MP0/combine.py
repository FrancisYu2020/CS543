import numpy as np
from PIL import Image
import os

for root, dirs, files in os.walk('shredded-images'):
    break

for dir in dirs:
    for subroot, subdirs, subfiles in os.walk(root + '/' + dir):
        break
    max_hight = 0
    total_width = 0
    for file in subfiles:
        im = Image.open(subroot + '/' + file)
        width, height = im.size
        if height > max_hight:
            max_hight = height
        total_width += width
    whole_array = np.zeros((max_hight, total_width, 3))
    end = 0
    for file in subfiles:
        shred = Image.open(subroot + '/' + file)
        shred_width, shred_height = shred.size
        # print(shred.size)
        shred_array = np.array(shred)
        # print(shred_array.shape)
        whole_array[:shred_height, end:end + shred_width] = shred_array
        end += shred_width
    # print(whole_array)
    save_image = Image.fromarray(whole_array.astype(np.uint8))
    save_image.save(dir + '.png')
