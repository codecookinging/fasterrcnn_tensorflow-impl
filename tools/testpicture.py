# -*- coding: utf-8 -*-

import os
import os.path
import numpy as np
import xml.etree.ElementTree as xmlET
from PIL import Image, ImageDraw
import pickle 

test_file = 'test.txt'
file_path_img = '../data/VOCdevkit2007/VOC2007/JPEGImages'
save_file_path = '../testimage'

with open(test_file) as f:
    image_index = [x.strip  () for x in f.readlines()]

f = open('detections.pkl','rb')  
info = pickle.load(f)


dets = info[1]
num = 0
for idx in range(len(dets)):
    if len(dets[idx]) == 0:
        continue
    img = Image.open(os.path.join(file_path_img, image_index[idx] + '.jpg')) 
    draw = ImageDraw.Draw(img)
    for i in range(len(dets[idx])):
        box = dets[idx][i]
        draw.rectangle([int(np.round(float(box[0]))), int(np.round(float(box[1]))), 
                    int(np.round(float(box[2]))), int(np.round(float(box[3])))], outline=(255, 0, 0))
    img.save(os.path.join(save_file_path, image_index[idx] + '.jpg'))
    