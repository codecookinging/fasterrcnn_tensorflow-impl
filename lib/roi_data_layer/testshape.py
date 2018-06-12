# -*- coding: utf-8 -*-
import numpy as np

from lib.datasets.pascal_voc import pascal_voc



ims=np.array([[4,6],[5,8],[4,10]])

num_images=ims.shape[0]


print(num_images)






print(ims.shape)
#[[400,600],[500,800],[456,1000]]
max_shape = np.array([im for im in ims]).max(axis=0) #取出所有图片的最大尺寸

blob = np.zeros((num_images, max_shape[0], max_shape[1], 3), #四维张量，
                  dtype=np.float32)



print(max_shape[0])
print(max_shape[1])
#print(blob)