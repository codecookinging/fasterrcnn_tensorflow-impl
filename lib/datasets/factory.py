# -*- coding: utf-8 -*-

# --------------------------------------------------------
#writen by saijunz

#此函数根据数据库的名字来获取数据库  比如 常见的 数据库 voc2007 voc2012 coco 等
#本函数目前只保留 voc 数据库
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

__sets = {}

import PIL
import numpy as np
from lib.datasets.pascal_voc import pascal_voc
from lib.model.config import cfg


# Set up voc_<year>_<split> 
for year in ['2007', '2012']:
  for split in ['train', 'val', 'trainval', 'test']:
    name = 'voc_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))


'''
# Set up coco_2014_<split>
for year in ['2014']:
  for split in ['train', 'val', 'minival', 'valminusminival', 'trainval']:
    name = 'coco_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up coco_2015_<split>
for year in ['2015']:
  for split in ['test', 'test-dev']:
    name = 'coco_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: coco(split, year))

'''
def get_imdb(name):
  """Get an imdb (image database) by name."""
  if name not in __sets:  #用字典来 存储 数据  key=name类似于 'voc_2007_train'这种形式 value 调用pascal 来产生
      
    raise KeyError('Unknown dataset: {}'.format(name))  #如果传入的 name 不存在 就报错
  return __sets[name]()  #返回对应的数据库


def list_imdbs():
  """List all registered imdbs."""
  return list(__sets.keys())  #字典的键值 就是所有数据库的名字


if __name__ == '__main__': 
    imdb_name='voc_2007_trainval'
    imdb = get_imdb(imdb_name)
    imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)  #设置gt 的方法
    roidb = imdb.roidb
    #print(len(roidb))
    #print(roidb[0])
    #imdb.image_path_at(i) 是返回第i张图片的路径
    sizes = [PIL.Image.open(imdb.image_path_at(i)).size #图片的大小用list 装起来
             for i in range(imdb.num_images)]
    

    
    for i in range(0,4):
        
        gt_overlaps = roidb[i]['gt_overlaps'].toarray()
        print(gt_overlaps)
        
        max_overlaps=gt_overlaps.argmax(axis=1) #最大的那个类下标
        print(max_overlaps)
        max_classes = gt_overlaps.max(axis=1) #最大的那个
        print(max_classes)
        
        
        
        
        