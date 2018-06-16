# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen

# 一张图片是一个minibatch 
#图片 短边取600 长边取1000 进行缩放，不够的补0，然后保存缩放尺度，将相应的anchor 也进行相同尺度的缩放
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network."""


import numpy as np
import numpy.random as npr
import cv2
from lib.model.config import cfg
from lib.utils.blob import prep_im_for_blob, im_list_to_blob

def get_minibatch(roidb, num_classes): #首先给每个图片进行缩放，后面给图片进行填充
  """Given a roidb, construct a minibatch sampled from it."""
  num_images = len(roidb)
  # Sample random scales to use for each image in this batch
  random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES),
                  size=num_images)
  assert(cfg.TRAIN.BATCH_SIZE % num_images == 0), \
    'num_images ({}) must divide BATCH_SIZE ({})'. \
    format(num_images, cfg.TRAIN.BATCH_SIZE)

  # Get the input image blob, formatted for caffe
  im_blob, im_scales = _get_image_blob(roidb, random_scale_inds)  

  blobs = {'data': im_blob}

  assert len(im_scales) == 1, "Single batch only"
  assert len(roidb) == 1, "Single batch only"
  
  # gt boxes: (x1, y1, x2, y2, cls)
  if cfg.TRAIN.USE_ALL_GT:
    # Include all ground truth boxes
    gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0] #需要类别不为0的gt 来训练
  else:
    # For the COCO ground truth boxes, exclude the ones that are ''iscrowd'' 
    gt_inds = np.where(roidb[0]['gt_classes'] != 0 & np.all(roidb[0]['gt_overlaps'].toarray() > -1.0, axis=1))[0] #属于前景的真实类别
  gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
  gt_boxes[:, 0:4] = roidb[0]['boxes'][gt_inds, :] * im_scales[0]
  gt_boxes[:, 4] = roidb[0]['gt_classes'][gt_inds]
  blobs['gt_boxes'] = gt_boxes
  blobs['im_info'] = np.array(
    [im_blob.shape[1], im_blob.shape[2], im_scales[0]],
    dtype=np.float32)

  return blobs

def _get_image_blob(roidb, scale_inds):
  """Builds an input blob from the images in the roidb at the specified
  scales.
  """
  num_images = len(roidb) 
  processed_ims = []
  im_scales = []
  for i in range(num_images):#读取图片   矩阵 
    im = cv2.imread(roidb[i]['image'])
    if roidb[i]['flipped']: #如果图片是水平对称的 那么将三维矩阵中第二维数据做对称操作
      im = im[:, ::-1, :]
    target_size = cfg.TRAIN.SCALES[scale_inds[i]] #确定选定的 缩放尺寸（最短边）的大小
    im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size, #调用blob 函数对图片进行缩放 并获取 scale
                    cfg.TRAIN.MAX_SIZE)
    im_scales.append(im_scale) #缩放系数保存在 list 里面
    processed_ims.append(im) #把三维数据作为一个元素放到list 里面去

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims) #填充后的图片 放入 blob 每张图片加入了 scale 在里面

  return blob, im_scales
