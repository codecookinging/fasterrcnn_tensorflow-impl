# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Zheqi He, Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------


from lib.model.train_val import get_training_roidb, train_net
from lib.model.config import cfg, cfg_from_file, get_output_dir, get_output_tb_dir
from lib.datasets.factory import get_imdb
import lib.datasets.imdb
import cv2
import pprint
import numpy as np
import sys
from lib.nets.vgg16 import vgg16
import tensorflow as tf

def get_roidb(imdb_name): #本函数根据数据库名字 获取roidb 
    
    imdb = get_imdb(imdb_name) #调用factory 的 imdb 然后根据 数据库 名字 调用 pascal_voc
    print('Loaded dataset `{:s}` for training'.format(imdb.name))
    imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)  #设置gt 的方法 
    print('Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD)) 
    roidb = get_training_roidb(imdb)
    
    return imdb,roidb
    
 

if __name__ == '__main__':
  
  # train set
  imdb, roidb = get_roidb('voc_2007_trainval') #训练集
  
  
  
  #print('{:d} roidb entries'.format(len(roidb)))
  pretrained_model='E:/fasterwrite/data/imagenet_weights/vgg16.ckpt'
  '''
  print(len(roidb))
  print (type(roidb[0]['image']))
  im=cv2.imread(roidb[0]['image']) 
  print(im)
  im=cv2.imread(roidb[6000]['image']) 
  print(im)
  '''

  
  weights_filename = 'default'
  output_dir = get_output_dir(imdb,weights_filename )  
  print('Output will be saved to `{:s}`'.format(output_dir))
 # tb_dir = get_output_tb_dir(imdb,weights_filename)
  
  
  net = vgg16()
  orgflip = cfg.TRAIN.USE_FLIPPED
  cfg.TRAIN.USE_FLIPPED = False
  valimdb,valroidb = get_roidb('voc_2007_test') #获取测试集数据
  #测试验证集数据
  #im=cv2.imread(valroidb[6000]['image'] 
  #print(im)
  
  print('{:d} validatio  n roidb entries'.format(len(valroidb)))
  cfg.TRAIN.USE_FLIPPED = orgflip
  #截止目前测试均通过
  
 
  
  #在源码基础上去除了很多冗余的东西。。但开始报错了
  train_net(net, imdb, roidb, valroidb, output_dir,
            pretrained_model=pretrained_model,
  max_iters=40000)
  
  # load network
  
 
    

