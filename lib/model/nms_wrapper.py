# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

from lib.model.config import cfg
#from lib.nms.gpu_nms import gpu_nms
#from lib.nms.cpu_nms import cpu_nms
from lib.nms import py_cpu_nms

def nms(dets, thresh, force_cpu=False):  #这个函数调用nms 函数 用来筛选proposal
  """Dispatch to either CPU or GPU NMS implementations."""
  '''
  if dets.shape[0] == 0:
    return []
  if cfg.USE_GPU_NMS and not force_cpu:
    return gpu_nms(dets, thresh, device_id=0)
  else:
    return cpu_nms(dets, thresh)
  '''
  return py_cpu_nms(dets, thresh)