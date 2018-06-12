# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen and Zheqi He
#此函数 将net 数据集合在一起 集大成者函数

# --------------------------------------------------------

from lib.model.config import cfg
import lib.roi_data_layer.roidb as rdl_roidb
from lib.roi_data_layer.layer import RoIDataLayer
from lib.utils.timer import Timer


import pickle
import numpy as np
import os
import sys
import glob
import time

import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

class SolverWrapper(object):
  """
    A wrapper class for the training process
    据作者的说法，这个类就是为了方便自己使用Python代码来控制训练过程中的相关东西  
    
    
  """

  def __init__(self, sess, network, imdb, roidb, valroidb, output_dir,pretrained_model):
    self.net = network #网络 vgg 或者 resnet
    self.imdb = imdb #数据库
    self.roidb = roidb  #region of insterest
    self.valroidb = valroidb #tensorboard 输出文件
    self.output_dir = output_dir  #结果输出文件
    #self.tbdir = tbdir
    # Simply put '_val' at the end to save the summaries from the validation set
    #self.tbvaldir = tbdir + '_val'
    #if not os.path.exists(self.tbvaldir):
      #os.makedirs(self.tbvaldir)
    self.pretrained_model = pretrained_model

  def snapshot(self, sess, iter): #这个函数做一些存储 存储
    net = self.net

    if not os.path.exists(self.output_dir):
      os.makedirs(self.output_dir)

    # Store the model snapshot
    filename = cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_{:d}'.format(iter) + '.ckpt' #__C.TRAIN.SNAPSHOT_PREFIX = 'res101_faster_rcnn' 默认输出模型
    filename = os.path.join(self.output_dir, filename) #存在output下面
    self.saver.save(sess, filename)  #存储tensor变量
    print('Wrote snapshot to: {:s}'.format(filename)) 
    return filename
    

  def from_snapshot(self, sess, sfile, nfile):
    print('Restoring model snapshots from {:s}'.format(sfile))
    self.saver.restore(sess, sfile)
    print('Restored.')
    # Needs to restore the other hyper-parameters/states for training, (TODO xinlei) I have
    # tried my best to find the random states so that it can be recovered exactly
    # However the Tensorflow state is currently not available
    with open(nfile, 'rb') as fid:
      st0 = pickle.load(fid)
      cur = pickle.load(fid)
      perm = pickle.load(fid)
      cur_val = pickle.load(fid)
      perm_val = pickle.load(fid)
      last_snapshot_iter = pickle.load(fid)

      np.random.set_state(st0)
      self.data_layer._cur = cur
      self.data_layer._perm = perm
      self.data_layer_val._cur = cur_val
      self.data_layer_val._perm = perm_val

    return last_snapshot_iter
 

    

  def get_variables_in_checkpoint_file(self, file_name):  #初始化模型 文件
    try:
      reader = pywrap_tensorflow.NewCheckpointReader(file_name)
      var_to_shape_map = reader.get_variable_to_shape_map()
      return var_to_shape_map 
    except Exception as e:  # pylint: disable=broad-except
      print(str(e))
      if "corrupted compressed block contents" in str(e):
        print("It's likely that your checkpoint file has been compressed "
              "with SNAPPY.")

  def construct_graph(self, sess):  #该函数构建计算图
    with sess.graph.as_default():
      # Set the random seed for tensorflow
      tf.set_random_seed(cfg.RNG_SEED) #随机种子
      # Build the main computation graph
      layers = self.net.create_architecture('TRAIN', self.imdb.num_classes,
                                            anchor_scales=cfg.ANCHOR_SCALES,
                                            anchor_ratios=cfg.ANCHOR_RATIOS)
      # Define the loss
      loss = layers['total_loss']
      # Set learning rate and momentum
      lr = tf.Variable(cfg.TRAIN.LEARNING_RATE, trainable=False) #学习率
      self.optimizer = tf.train.MomentumOptimizer(lr, cfg.TRAIN.MOMENTUM) #梯度优化器

      # Compute the gradients with regard to the loss
      gvs = self.optimizer.compute_gradients(loss)
      # Double the gradient of the bias if set
      if cfg.TRAIN.DOUBLE_BIAS:
        final_gvs = []
        with tf.variable_scope('Gradient_Mult') as scope:
          for grad, var in gvs:
            scale = 1.
            if cfg.TRAIN.DOUBLE_BIAS and '/biases:' in var.name:
              scale *= 2.
            if not np.allclose(scale, 1.0):
              grad = tf.multiply(grad, scale)
            final_gvs.append((grad, var))
        train_op = self.optimizer.apply_gradients(final_gvs)
      else:
        train_op = self.optimizer.apply_gradients(gvs)

      # We will handle the snapshots ourselves
      self.saver = tf.train.Saver(max_to_keep=100000)
      # Write the train and validation information to tensorboard
      #self.writer = tf.summary.FileWriter(self.tbdir, sess.graph)
      #self.valwriter = tf.summary.FileWriter(self.tbvaldir)
      print("构建网络通过")
    return lr, train_op

  def find_previous(self):
    sfiles = os.path.join(self.output_dir, cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_*.ckpt.meta')
    sfiles = glob.glob(sfiles)
    sfiles.sort(key=os.path.getmtime)
    # Get the snapshot name in TensorFlow
    redfiles = []
    for stepsize in cfg.TRAIN.STEPSIZE:
      redfiles.append(os.path.join(self.output_dir, 
                      cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_{:d}.ckpt.meta'.format(stepsize+1)))
    sfiles = [ss.replace('.meta', '') for ss in sfiles if ss not in redfiles]

    nfiles = os.path.join(self.output_dir, cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_*.pkl')
    nfiles = glob.glob(nfiles)
    nfiles.sort(key=os.path.getmtime)
    redfiles = [redfile.replace('.ckpt.meta', '.pkl') for redfile in redfiles]
    nfiles = [nn for nn in nfiles if nn not in redfiles]

    lsf = len(sfiles)
    assert len(nfiles) == lsf

    return lsf, nfiles, sfiles

  def initialize(self, sess):
    # Initial file lists are empty
    np_paths = []
    ss_paths = []
    # Fresh train directly from ImageNet weights
    print('Loading initial model weights from {:s}'.format(self.pretrained_model))
    variables = tf.global_variables()
    # Initialize all variables first
    sess.run(tf.variables_initializer(variables, name='init'))
    var_keep_dic = self.get_variables_in_checkpoint_file(self.pretrained_model) #把预训练网络参数 拿出来 
    # Get the variables to restore, ignoring the variables to fix
    variables_to_restore = self.net.get_variables_to_restore(variables, var_keep_dic)

    restorer = tf.train.Saver(variables_to_restore)
    restorer.restore(sess, self.pretrained_model)
    print('Loaded.')
    # Need to fix the variables before loading, so that the RGB weights are changed to BGR
    # For VGG16 it also changes the convolutional weights fc6 and fc7 to
    # fully connected weights
    self.net.fix_variables(sess, self.pretrained_model)
    print('Fixed.')
    last_snapshot_iter = 0
    rate = cfg.TRAIN.LEARNING_RATE
    stepsizes = list(cfg.TRAIN.STEPSIZE)

    return rate, last_snapshot_iter, stepsizes, np_paths, ss_paths

  def restore(self, sess, sfile, nfile):
    # Get the most recent snapshot and restore
    np_paths = [nfile]
    ss_paths = [sfile]
    # Restore model from snapshots
    last_snapshot_iter = self.from_snapshot(sess, sfile, nfile)
    # Set the learning rate
    rate = cfg.TRAIN.LEARNING_RATE
    stepsizes = []
    for stepsize in cfg.TRAIN.STEPSIZE:
      if last_snapshot_iter > stepsize:
        rate *= cfg.TRAIN.GAMMA
      else:
        stepsizes.append(stepsize)

    return rate, last_snapshot_iter, stepsizes, np_paths, ss_paths

  def remove_snapshot(self, np_paths, ss_paths):
    to_remove = len(np_paths) - cfg.TRAIN.SNAPSHOT_KEPT
    for c in range(to_remove):
      nfile = np_paths[0]
      os.remove(str(nfile))
      np_paths.remove(nfile)

    to_remove = len(ss_paths) - cfg.TRAIN.SNAPSHOT_KEPT
    for c in range(to_remove):
      sfile = ss_paths[0]
      # To make the code compatible to earlier versions of Tensorflow,
      # where the naming tradition for checkpoints are different
      if os.path.exists(str(sfile)):
        os.remove(str(sfile))
      else:
        os.remove(str(sfile + '.data-00000-of-00001'))
        os.remove(str(sfile + '.index'))
      sfile_meta = sfile + '.meta'
      os.remove(str(sfile_meta))
      ss_paths.remove(sfile)

  def train_model(self, sess, max_iters):
    # Build data layers for both training and validation set
    self.data_layer = RoIDataLayer(self.roidb, self.imdb.num_classes)
    self.data_layer_val = RoIDataLayer(self.valroidb, self.imdb.num_classes, random=True)

    # Construct the computation graph
    lr, train_op = self.construct_graph(sess)

    # Find previous snapshots if there is any to restore from
    lsf, nfiles, sfiles = self.find_previous()

    # Initialize the variables or restore them from the last snapshot
    if lsf == 0:
      rate, last_snapshot_iter, stepsizes, np_paths, ss_paths = self.initialize(sess)
    else:
      rate, last_snapshot_iter, stepsizes, np_paths, ss_paths = self.restore(sess, 
                                                                            str(sfiles[-1]), 
                                                                            str(nfiles[-1]))
    timer = Timer()
    iter = last_snapshot_iter + 1
    last_summary_time = time.time()
    # Make sure the lists are not empty
    stepsizes.append(max_iters)
    stepsizes.reverse()
    next_stepsize = stepsizes.pop()
    while iter < max_iters + 1:
      # Learning rate
      if iter == next_stepsize + 1:
        # Add snapshot here before reducing the learning rate
        self.snapshot(sess, iter)
        rate *= cfg.TRAIN.GAMMA
        sess.run(tf.assign(lr, rate))
        next_stepsize = stepsizes.pop()

      timer.tic()
      # Get training data, one batch at a time
      blobs = self.data_layer.forward()

      now = time.time()
      if iter == 1 or now - last_summary_time > cfg.TRAIN.SUMMARY_INTERVAL:
        # Compute the graph with summary
        rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, total_loss, summary = \
          self.net.train_step_with_summary(sess, blobs, train_op)
        self.writer.add_summary(summary, float(iter))
        # Also check the summary on the validation set
        blobs_val = self.data_layer_val.forward()
        summary_val = self.net.get_summary(sess, blobs_val)
        self.valwriter.add_summary(summary_val, float(iter))
        last_summary_time = now
      else:
        # Compute the graph without summary
        rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, total_loss = \
          self.net.train_step(sess, blobs, train_op)
      timer.toc()

      # Display training information
      if iter % (cfg.TRAIN.DISPLAY) == 0:
        print('iter: %d / %d, total loss: %.6f\n >>> rpn_loss_cls: %.6f\n '
              '>>> rpn_loss_box: %.6f\n >>> loss_cls: %.6f\n >>> loss_box: %.6f\n >>> lr: %f' % \
              (iter, max_iters, total_loss, rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, lr.eval()))
        print('speed: {:.3f}s / iter'.format(timer.average_time))

      # Snapshotting
      if iter % cfg.TRAIN.SNAPSHOT_ITERS == 0:
        last_snapshot_iter = iter
        ss_path, np_path = self.snapshot(sess, iter)
        np_paths.append(np_path)
        ss_paths.append(ss_path)

        # Remove the old snapshots if there are too many
        if len(np_paths) > cfg.TRAIN.SNAPSHOT_KEPT:
          self.remove_snapshot(np_paths, ss_paths)

      iter += 1

    if last_snapshot_iter != iter - 1:
      self.snapshot(sess, iter - 1)

    self.writer.close()
    self.valwriter.close()


def get_training_roidb(imdb):
  """Returns a roidb (Region of Interest database) for use in training."""
  if cfg.TRAIN.USE_FLIPPED:
    print('Appending horizontally-flipped training examples...') #这个函数扩增数据集 在imdb 基础上进一步封装 
    imdb.append_flipped_images()
    print('done')

  print('Preparing training data...') 
  rdl_roidb.prepare_roidb(imdb)
  print('done')

  return imdb.roidb


def filter_roidb(roidb):
  """Remove roidb entries that have no usable RoIs."""
  print("调用filter 了开始")
  def is_valid(entry):
    # Valid images have:
    #   (1) At least one foreground RoI OR
    #   (2) At least one background RoI
    overlaps = entry['max_overlaps']
    # find boxes with sufficient overlap
    fg_inds = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]  #前景为overlap大于某个值
    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) &
                       (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
    # image is only valid if such boxes exist
    valid = len(fg_inds) > 0 or len(bg_inds) > 0
    return valid

  num = len(roidb)
  filtered_roidb = [entry for entry in roidb if is_valid(entry)]
  num_after = len(filtered_roidb)
  print('Filtered {} roidb entries: {} -> {}'.format(num - num_after,
                                                     num, num_after))
  return filtered_roidb


def train_net(network, imdb, roidb, valroidb, output_dir,
              pretrained_model=None, max_iters=40000):
             
  """Train a Faster R-CNN network."""
  #roidb = filter_roidb(roidb) #这是训练集
  #valroidb = filter_roidb(valroidb) #这是验证集

  tfconfig = tf.ConfigProto(allow_soft_placement=True)
  tfconfig.gpu_options.allow_growth = True

  with tf.Session(config=tfconfig) as sess:
    sw = SolverWrapper(sess, network, imdb, roidb, valroidb, output_dir,
                       pretrained_model=pretrained_model)
    print('Solving...')
    sw.train_model(sess, max_iters)
    print('done solving')
