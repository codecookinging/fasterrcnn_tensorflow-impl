# -*- coding: utf-8 -*-

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by saijunz
# --------------------------------------------------------


import numpy as np
import numpy.random as npr

import cv2
from lib.model.config import cfg
from lib.utils.blob import prep_im_for_blob, im_list_to_blob




random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES),  ## The scale is the pixel size of an image's shortest side 600 
                  size=3000) #根据scales数量为 每张图片生成一个scales 索引



print(random_scale_inds)


 