# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# 这个函数把一系列图片 转化为 网络输入 （包括 图片缩放 填充）
# Written by saijunz
# --------------------------------------------------------

"""Blob helper functions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2


def im_list_to_blob(ims):
  """Convert a list of images into a network input.

  Assumes images are already prepared (means subtracted, BGR order, ...). #确保图片之前经过这些预处理 (到这里为止图片已经缩放完毕) 比如 长边 最大为1000 短边最大为500 接下来进行填充
  """
  max_shape = np.array([im.shape for im in ims]).max(axis=0) #取出所有图片的最大尺寸 长宽都取最大
  num_images = len(ims) #图片个数
  blob = np.zeros((num_images, max_shape[0], max_shape[1], 3), #四维张量，小图片边缘就用0来填充
                  dtype=np.float32)
  for i in range(num_images): 
    im = ims[i]
    blob[i, 0:im.shape[0], 0:im.shape[1], :] = im

  return blob    


def prep_im_for_blob(im, pixel_means, target_size, max_size): #@该方法主要是求取图像的缩放比例，然后将图像resize
  """Mean subtract and scale an image for use in a blob."""
  im = im.astype(np.float32, copy=False)
  im -= pixel_means  #为啥要减去像素均值呢
  im_shape = im.shape
  im_size_min = np.min(im_shape[0:2]) #图片最短边
  im_size_max = np.max(im_shape[0:2]) #图片最长边
  im_scale = float(target_size) / float(im_size_min) #尺度   默认按短边来缩放 比如 标准是 600 1000    如果缩放后长边 大于1000 那就按长边缩放
  # Prevent the biggest axis from being more than MAX_SIZE
  if np.round(im_scale * im_size_max) > max_size:   
    im_scale = float(max_size) / float(im_size_max)
  im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                  interpolation=cv2.INTER_LINEAR)

  return im, im_scale
