# --------------------------------------------------------
# Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by saijunz 
#原谅我 之前没看懂，原来 该函数只是对每种函数写了两个版本
#这个函数 需要结合之前网络层传过来的 rpn_score 以及回归参数 来做
 #blobs['im_info'] = np.array(
#[im_blob.shape[1], im_blob.shape[2], im_scales[0]],
#dtype=np.float32)



# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from lib.model.config import cfg
from lib.model.bbox_transform import bbox_transform_inv, clip_boxes, bbox_transform_inv_tf, clip_boxes_tf
from lib.model.nms_wrapper import nms

def proposal_layer(rpn_cls_prob, rpn_bbox_pred, im_info, cfg_key, _feat_stride, anchors, num_anchors):
  """A simplified version compared to fast/er RCNN
     For details please see the technical report
  """
  if type(cfg_key) == bytes:
      cfg_key = cfg_key.decode('utf-8')
  pre_nms_topN = cfg[cfg_key].RPN_PRE_NMS_TOP_N
  post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
  nms_thresh = cfg[cfg_key].RPN_NMS_THRESH

  # Get the scores and bounding boxes
  scores = rpn_cls_prob[:, :, :, num_anchors:]#每个anchor 有打分
  rpn_bbox_pred = rpn_bbox_pred.reshape((-1, 4)) 
  scores = scores.reshape((-1, 1))
  proposals = bbox_transform_inv(anchors, rpn_bbox_pred) #通过这个函数 把 anchor 转化成 proposal?   根据anchor 与 rpn_bbox 返回一个预测的 proposal
  proposals = clip_boxes(proposals, im_info[:2]) 

  # Pick the top region proposals
  order = scores.ravel().argsort()[::-1]    #筛选出 top_N的 proposal
  if pre_nms_topN > 0:
    order = order[:pre_nms_topN]
  proposals = proposals[order, :]
  scores = scores[order]

  # Non-maximal suppression
  keep = nms(np.hstack((proposals, scores)), nms_thresh)  #极大抑制算法 继续筛选proposal  NMS threshold used on RPN proposals=0.7 重合度大于0.7视为同一类

  # Pick th top region proposals after NMS   
  if post_nms_topN > 0:
    keep = keep[:post_nms_topN]
  proposals = proposals[keep, :]
  scores = scores[keep]  #挑选 排序最好的 几个proposal 

  # Only support single image as input
  batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)  
  blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False))) #返回 proposal 列表

  return blob, scores


def proposal_layer_tf(rpn_cls_prob, rpn_bbox_pred, im_info, cfg_key, _feat_stride, anchors, num_anchors): #通过 anchor bbox_pred 产出 proposals
  if type(cfg_key) == bytes:
    cfg_key = cfg_key.decode('utf-8')
  pre_nms_topN = cfg[cfg_key].RPN_PRE_NMS_TOP_N
  post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
  nms_thresh = cfg[cfg_key].RPN_NMS_THRESH

  # Get the scores and bounding boxes
  scores = rpn_cls_prob[:, :, :, num_anchors:]
  scores = tf.reshape(scores, shape=(-1,))
  rpn_bbox_pred = tf.reshape(rpn_bbox_pred, shape=(-1, 4))

  proposals = bbox_transform_inv_tf(anchors, rpn_bbox_pred)
  proposals = clip_boxes_tf(proposals, im_info[:2])

  # Non-maximal suppression
  indices = tf.image.non_max_suppression(proposals, scores, max_output_size=post_nms_topN, iou_threshold=nms_thresh)  
  ##NMS 对检测得到的全部 boxes 进行局部的最大搜索，以搜索某邻域范围内的最大值，
  #从而滤出一部分 boxes，提升最终的检测精度.
  boxes = tf.gather(proposals, indices)
  boxes = tf.to_float(boxes)
  scores = tf.gather(scores, indices)
  scores = tf.reshape(scores, shape=(-1, 1))

  # Only support single image as input
  batch_inds = tf.zeros((tf.shape(indices)[0], 1), dtype=tf.float32)
  blob = tf.concat([batch_inds, boxes], 1)

  return blob, scores


