# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Zheqi He, Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------


from lib.model.train_val import get_training_roidb, train_net
from lib.model.config import cfg, cfg_from_file, get_output_dir, get_output_tb_dir
from lib.datasets.factory import get_imdb
import lib.datasets.imdb

import pprint
import numpy as np
import sys
from lib.nets.vgg16 import vgg16
import tensorflow as tf

def combined_roidb(imdb_names):
  """
  Combine multiple roidbs
  """

  def get_roidb(imdb_name):
    imdb = get_imdb(imdb_name) #调用factory 的 imdb 然后根据 数据库 名字 调用 pascal_voc
    print('Loaded dataset `{:s}` for training'.format(imdb.name))
    imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)  #设置gt 的方法 
    print('Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD)) 
    roidb = get_training_roidb(imdb) #根据数据库名字 来获取 region proposal
    return roidb   

  roidbs = [get_roidb(s) for s in imdb_names.split('+')] #这种是预防 两个数据库的存在吗
  roidb = roidbs[0]
  if len(roidbs) > 1:
    for r in roidbs[1:]:
      roidb.extend(r)
    tmp = get_imdb(imdb_names.split('+')[1])
    imdb = datasets.imdb.imdb(imdb_names, tmp.classes)
  else:
    imdb = get_imdb(imdb_names)
  return imdb, roidb


if __name__ == '__main__':
  
  # train set
  imdb, roidb = combined_roidb('voc_2007_trainval') #训练集
  print('{:d} roidb entries'.format(len(roidb)))
  
  pretrained_model='E:/fasterwrite/data/imagenet_weights/vgg16.ckpt'
  
  
  
  
  
  weights_filename = 'default'
  output_dir = get_output_dir(imdb,weights_filename )  
  print('Output will be saved to `{:s}`'.format(output_dir))
 # tb_dir = get_output_tb_dir(imdb,weights_filename)
  
  
  net = vgg16()
  orgflip = cfg.TRAIN.USE_FLIPPED
  cfg.TRAIN.USE_FLIPPED = False
  valroidb = combined_roidb('voc_2007_test') #测试集
  print('{:d} validatio  n roidb entries'.format(len(valroidb)))
  cfg.TRAIN.USE_FLIPPED = orgflip
  

   
 
  train_net(net, imdb, roidb, valroidb, output_dir,
            pretrained_model=pretrained_model,
  max_iters=40000)
 
  # load network
  
 
    

