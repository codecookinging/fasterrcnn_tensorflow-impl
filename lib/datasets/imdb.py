# -*- coding: utf-8 -*-

# --------------------------------------------------------
# 这个类应该是针对所有数据库的通用类，后面的 具体数据库处理类 都要继承它 实现它的方法
# written by zsj 2018/06/03

#本类有几个作用：
#(1) 根据数据库提供的文档 生成 ground_truth 即 真实的检测对象  这里用个字典存储 self._obj_proposer = 'gt'
#(2) 根据 结合数据库数据 建立索引系统 ，明确每个变量的index 
#       gt_roidb  的返回值
#return {'boxes': boxes,
#            'gt_classes': gt_classes,
#            'gt_overlaps': overlaps,
 #           'flipped': False,
#            'seg_areas': seg_areas}



# --------------------------------------------------------


import os
import os.path as osp
import PIL
import sys
 
from lib.utils.cython_bbox import bbox_overlaps
import numpy as np
import scipy.sparse
from lib.model.config import cfg


class imdb(object):
  """Image database."""

  def __init__(self, name, classes=None):
    self._name = name
    self._num_classes = 0
    if not classes:
      self._classes = []
    else:
      self._classes = classes
    self._image_index = []
    self._obj_proposer = 'gt'
    self._roidb = None
    self._roidb_handler = self.default_roidb
    # Use this dict for storing dataset specific config options
    self.config = {}

  @property   #下面这些函数 目的是获取 图片类的 属性
  def name(self):  
    return self._name  #获取数据库名称

  @property
  def num_classes(self):
    return len(self._classes)#获取 类个数 也即是 待检测物体的种类个数

  @property
  def classes(self):
    return self._classes #获取某个类

  @property
  def image_index(self):
    return self._image_index #获取图片索引

  @property
  def roidb_handler(self):
    return self._roidb_handler #处理 gt 区域的函数

  @roidb_handler.setter
  def roidb_handler(self, val):
    self._roidb_handler = val

  def set_proposal_method(self, method):
    method = eval('self.' + method + '_roidb')
    self.roidb_handler = method

  @property
  def roidb(self):
    # A roidb is a list of dictionaries, each with the following keys:
    #   boxes #box 存储 xmin ymin xmax ymax 坐标值
    #   gt_overlaps #存储 
    #   gt_classes #存储 对应类别
    #   flipped #是否翻转
    if self._roidb is not None:
      return self._roidb
    self._roidb = self.roidb_handler()
    return self._roidb

  @property
  def cache_path(self):
    cache_path = osp.abspath(osp.join(cfg.DATA_DIR, 'cache'))  #这里面的 cfg.DATA_DIR 即是存储 voc数据库的路径 类似于 /voc2007/cachepath
    
    if not os.path.exists(cache_path):
      os.makedirs(cache_path) #没有的话新建Cache文件夹
    return cache_path

  @property
  def num_images(self):
    return len(self.image_index)

  def image_path_at(self, i):
    raise NotImplementedError

  def default_roidb(self):
    raise NotImplementedError

  def evaluate_detections(self, all_boxes, output_dir=None):
    """
    all_boxes is a list of length number-of-classes.
    Each list element is a list of length number-of-images.
    Each of those list elements is either an empty list []
    or a numpy array of detection.
    all_boxes[class][image] = [] or np.array of shape #dets x 5
    """
    raise NotImplementedError

  def _get_widths(self):
    return [PIL.Image.open(self.image_path_at(i)).size[0]
            for i in range(self.num_images)]

  def append_flipped_images(self):  #数据扩增一倍 加入了 flipped image 的index  ,利用已有的宽度 以及已有的box 把 原来的roidb 完全拷贝一份
    num_images = self.num_images  
    widths = self._get_widths() #获取
    for i in range(num_images):
      boxes = self.roidb[i]['boxes'].copy()
      oldx1 = boxes[:, 0].copy() #xmin
      oldx2 = boxes[:, 2].copy() #xmax
      boxes[:, 0] = widths[i] - oldx2 - 1  #xmin+xmax+1=width
      boxes[:, 2] = widths[i] - oldx1 - 1
      assert (boxes[:, 2] >= boxes[:, 0]).all()
      entry = {'boxes': boxes,
               'gt_overlaps': self.roidb[i]['gt_overlaps'], 
               'gt_classes': self.roidb[i]['gt_classes'],
               'flipped': True}
      self.roidb.append(entry)
    self._image_index = self._image_index * 2

  def evaluate_recall(self, candidate_boxes=None, thresholds=None,
                      area='all', limit=None):
    """Evaluate detection proposal recall metrics. #评估检测的召回率
    Returns:
        results: dictionary of results with keys   #字典
            'ar': average recall                   #平均召回率
            'recalls': vector recalls at each IoU overlap threshold #每个IOU 重叠阈值
            'thresholds': vector of IoU overlap thresholds  #
            'gt_overlaps': vector of all ground-truth overlaps # 所有groud_truth IOU重叠阈值
    """
    # Record max overlap value for each gt box
    # Return vector of overlap values
    areas = {'all': 0, 'small': 1, 'medium': 2, 'large': 3,
             '96-128': 4, '128-256': 5, '256-512': 6, '512-inf': 7}
    area_ranges = [[0 ** 2, 1e5 ** 2],  # all  肯定包含所有area
                   [0 ** 2, 32 ** 2],  # small  小面积情况
                   [32 ** 2, 96 ** 2],  # medium 中等面积
                   [96 ** 2, 1e5 ** 2],  # large  大面积
                   [96 ** 2, 128 ** 2],  # 96-128 
                   [128 ** 2, 256 ** 2],  # 128-256
                   [256 ** 2, 512 ** 2],  # 256-512
                   [512 ** 2, 1e5 ** 2],  # 512-inf
                   ]
    assert area in areas, 'unknown area range: {}'.format(area)
    area_range = area_ranges[areas[area]]
    gt_overlaps = np.zeros(0)
    num_pos = 0
    for i in range(self.num_images):
      # Checking for max_overlaps == 1 avoids including crowd annotations
      # (...pretty hacking :/)
      max_gt_overlaps = self.roidb[i]['gt_overlaps'].toarray().max(axis=1)
      gt_inds = np.where((self.roidb[i]['gt_classes'] > 0) &
                         (max_gt_overlaps == 1))[0]
      gt_boxes = self.roidb[i]['boxes'][gt_inds, :]
      gt_areas = self.roidb[i]['seg_areas'][gt_inds]
      valid_gt_inds = np.where((gt_areas >= area_range[0]) &
                               (gt_areas <= area_range[1]))[0]
      gt_boxes = gt_boxes[valid_gt_inds, :]
      num_pos += len(valid_gt_inds)

      if candidate_boxes is None:
        # If candidate_boxes is not supplied, the default is to use the
        # non-ground-truth boxes from this roidb
        non_gt_inds = np.where(self.roidb[i]['gt_classes'] == 0)[0]
        boxes = self.roidb[i]['boxes'][non_gt_inds, :]
      else:
        boxes = candidate_boxes[i]
      if boxes.shape[0] == 0:
        continue
      if limit is not None and boxes.shape[0] > limit:
        boxes = boxes[:limit, :]

      overlaps = bbox_overlaps(boxes.astype(np.float),
                               gt_boxes.astype(np.float))

      _gt_overlaps = np.zeros((gt_boxes.shape[0]))
      for j in range(gt_boxes.shape[0]):
        # find which proposal box maximally covers each gt box
        argmax_overlaps = overlaps.argmax(axis=0)
        # and get the iou amount of coverage for each gt box
        max_overlaps = overlaps.max(axis=0)
        # find which gt box is 'best' covered (i.e. 'best' = most iou)
        gt_ind = max_overlaps.argmax()
        gt_ovr = max_overlaps.max()
        assert (gt_ovr >= 0)
        # find the proposal box that covers the best covered gt box
        box_ind = argmax_overlaps[gt_ind]
        # record the iou coverage of this gt box
        _gt_overlaps[j] = overlaps[box_ind, gt_ind]
        assert (_gt_overlaps[j] == gt_ovr)
        # mark the proposal box and the gt box as used
        overlaps[box_ind, :] = -1
        overlaps[:, gt_ind] = -1
      # append recorded iou coverage level
      gt_overlaps = np.hstack((gt_overlaps, _gt_overlaps))

    gt_overlaps = np.sort(gt_overlaps)
    if thresholds is None:
      step = 0.05
      thresholds = np.arange(0.5, 0.95 + 1e-5, step)
    recalls = np.zeros_like(thresholds)
    # compute recall for each iou threshold
    for i, t in enumerate(thresholds):
      recalls[i] = (gt_overlaps >= t).sum() / float(num_pos)
    # ar = 2 * np.trapz(recalls, thresholds)
    ar = recalls.mean()
    return {'ar': ar, 'recalls': recalls, 'thresholds': thresholds,
            'gt_overlaps': gt_overlaps}

  def create_roidb_from_box_list(self, box_list, gt_roidb): # 通过 gt_roidb产生 roidb
    assert len(box_list) == self.num_images, \
      'Number of boxes must match number of ground-truth images'   #每张图片对应一个 box_list 一个 box_list 有好几个 box 多物体识别
    roidb = []
    for i in range(self.num_images):
      boxes = box_list[i] 
      num_boxes = boxes.shape[0] #每个box_list的个数
      overlaps = np.zeros((num_boxes, self.num_classes), dtype=np.float32) #计算每个 box 与 每个ground_truth 的重叠面积

      if gt_roidb is not None and gt_roidb[i]['boxes'].size > 0: #groudth 存在 且 box 有尺寸
        gt_boxes = gt_roidb[i]['boxes']                #将 赋值给roidb
        gt_classes = gt_roidb[i]['gt_classes']         # 类别赋值
        gt_overlaps = bbox_overlaps(boxes.astype(np.float),  #roidb 是在 gt_roidb  基础上 与初始化的 anchor 形成的 overlaps
                                    gt_boxes.astype(np.float))
        argmaxes = gt_overlaps.argmax(axis=1) 
        maxes = gt_overlaps.max(axis=1)
        I = np.where(maxes > 0)[0]
        overlaps[I, gt_classes[argmaxes[I]]] = maxes[I]

      overlaps = scipy.sparse.csr_matrix(overlaps)
      roidb.append({
        'boxes': boxes,
        'gt_classes': np.zeros((num_boxes,), dtype=np.int32),
        'gt_overlaps': overlaps,
        'flipped': False,
        'seg_areas': np.zeros((num_boxes,), dtype=np.float32),
      })
    return roidb

  @staticmethod
  def merge_roidbs(a, b):   #合并两个 roidb
    assert len(a) == len(b)
    for i in range(len(a)):
      a[i]['boxes'] = np.vstack((a[i]['boxes'], b[i]['boxes']))  
      a[i]['gt_classes'] = np.hstack((a[i]['gt_classes'],
                                      b[i]['gt_classes']))
      a[i]['gt_overlaps'] = scipy.sparse.vstack([a[i]['gt_overlaps'],
                                                 b[i]['gt_overlaps']])
      a[i]['seg_areas'] = np.hstack((a[i]['seg_areas'],
                                     b[i]['seg_areas']))
    return a

