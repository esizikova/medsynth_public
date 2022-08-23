# ------------------------------------------------------------------------------
# Copyright (c) Tencent
# Licensed under the GPLv3 License.
# Created by Kai Ma (makai0324@gmail.com)
# ------------------------------------------------------------------------------

import argparse
from lib.config.config import cfg_from_yaml, cfg, merge_dict_and_yaml, print_easy_dict
from lib.dataset.factory import get_dataset
from lib.model.factory import get_model
import copy
import torch
import time
import os
import random

from lib.dataset.baseDataSet import Base_DataSet
import numpy as np
import cv2

# where to place LIDC dataset
TBX_PATH='/scratch/es5223/medsynth/data/TBX11K/'
# where to save model
MODEL_DEST_PATH = '/scratch/es5223/medsynth/clean/X2CT/3DGAN/projection_model_experiments/1_norm/'
 # where pretrained model was saved
MODEL_SOURCE_PATH = '/scratch/es5223/medsynth/clean/X2CT/3DGAN/save_models/singleView_CTGAN/LIDC256/d2_singleview2500_256/'

class AlignDataSet_SingleImage(Base_DataSet):
  '''
  DataSet For unaligned data
  '''
  def __init__(self, opt):
    super(AlignDataSet_SingleImage, self).__init__()
    self.opt = opt
    self.data_augmentation = self.opt.data_augmentation(opt)

    flList = open(TBX_PATH+'lists/all_train.txt','r')
    lines = flList.readlines()
    flList.close()
    self.image_paths = [line.strip() for line in lines]
    random.shuffle(self.image_paths)
    self.image_paths = self.image_paths[0:229*4]
  @property
  def name(self):
    return 'AlignDataSet_SingleImage'

  @property
  def get_data_path(self):
    path = os.path.join(self.opt.dataroot)
    return path

  @property
  def num_samples(self):
    return len(self.image_paths) # self.dataset_size

  '''
  generate batch
  '''
  def pull_item(self, index):
    ct_data = np.zeros((256,256,256))

    X_IMG_PATH = TBX_PATH+'/imgs/' + self.image_paths[index]
    x_ray1_0 = cv2.imread(X_IMG_PATH, cv2.IMREAD_GRAYSCALE)    
    x_ray1_resized = cv2.resize(x_ray1_0, (256,256), interpolation = cv2.INTER_CUBIC)
    x_ray1 = np.expand_dims(x_ray1_resized,0)
    
    file_path = None
    # Data Augmentation
    ct, xray1 = self.data_augmentation([ct_data, x_ray1])

    return ct, xray1, file_path



def parse_args():
  parse = argparse.ArgumentParser(description='CTGAN')
  parse.add_argument('--data', type=str, default='', dest='data',
                     help='input data ')
  parse.add_argument('--tag', type=str, default='', dest='tag',
                     help='distinct from other try')
  parse.add_argument('--dataroot', type=str, default='', dest='dataroot',
                     help='input data root')
  parse.add_argument('--dataset', type=str, default='', dest='dataset',
                     help='Train or test or valid')
  parse.add_argument('--valid_dataset', type=str, default=None, dest='valid_dataset',
                     help='Train or test or valid')
  parse.add_argument('--datasetfile', type=str, default='', dest='datasetfile',
                     help='Train or test or valid file path')
  parse.add_argument('--valid_datasetfile', type=str, default='', dest='valid_datasetfile',
                     help='Train or test or valid file path')
  parse.add_argument('--ymlpath', type=str, default=None, dest='ymlpath',
                     help='config have been modified')
  parse.add_argument('--gpu', type=str, default='0,1', dest='gpuid',
                     help='gpu is split by ,')
  parse.add_argument('--dataset_class', type=str, default='align', dest='dataset_class',
                     help='Dataset class should select from align /')
  parse.add_argument('--model_class', type=str, default='simpleGan', dest='model_class',
                     help='Model class should select from simpleGan / ')
  parse.add_argument('--check_point', type=str, default=None, dest='check_point',
                     help='which epoch to load? ')
  parse.add_argument('--load_path', type=str, default=None, dest='load_path',
                     help='if load_path is not None, model will load from load_path')
  parse.add_argument('--latest', action='store_true', dest='latest',
                     help='set to latest to use latest cached model')
  parse.add_argument('--verbose', action='store_true', dest='verbose',
                     help='if specified, print more debugging information')
  parse.add_argument('--imageGAN', action='store_true', dest='imageGAN',
                     help='useImageGAN instead of patchGAN for training')
  parse.add_argument('--save_path', type=str, default=None, dest='save_path',
                     help='..savepath')
   
  args = parse.parse_args()
  return args

if __name__ == '__main__':
  args = parse_args()

  # check gpu
  if args.gpuid == '':
    args.gpu_ids = []
  else:
    if torch.cuda.is_available():
      split_gpu = str(args.gpuid).split(',')
      args.gpu_ids = [int(i) for i in split_gpu]
    else:
      print('There is no gpu!')
      exit(0)

  # check point
  if args.check_point is None:
    args.epoch_count = 1
  else:
    args.epoch_count = int(args.check_point) + 1

  # merge config with yaml
  if args.ymlpath is not None:
    cfg_from_yaml(args.ymlpath)
  # merge config with argparse
  opt = copy.deepcopy(cfg)
  opt = merge_dict_and_yaml(args.__dict__, opt)
  if args.save_path!='':
    opt.MODEL_SAVE_PATH = args.save_path
    
  #################################  
  ### hard code some options
  opt.epoch_count = 10 # number of epochs to train
    
  # pretrained model to load
  opt.check_point = 90
  opt.latest = None
  # where to save results
  opt.save_path = MODEL_DEST_PATH 
 
  # where pretrained model was saved
  opt.MODEL_SAVE_PATH = MODEL_SOURCE_PATH
  #################################


  print_easy_dict(opt)

  # add data_augmentation
  datasetClass, augmentationClass, dataTestClass, collateClass = get_dataset(opt.dataset_class)
  opt.data_augmentation = augmentationClass

  # valid dataset
  if args.valid_dataset is not None:
    valid_opt = copy.deepcopy(opt)
    valid_opt.data_augmentation = dataTestClass
    valid_opt.datasetfile = opt.valid_datasetfile


    valid_dataset = datasetClass(valid_opt)
    print('Valid DataSet is {}'.format(valid_dataset.name))
    valid_dataloader = torch.utils.data.DataLoader(
      valid_dataset,
      batch_size=1,
      shuffle=False,
      num_workers=int(valid_opt.nThreads),
      collate_fn=collateClass)
    valid_dataset_size = len(valid_dataloader)
    print('#validation images = %d' % valid_dataset_size)
  else:
    valid_dataloader = None

  # get dataset
  dataset = datasetClass(opt)
  print('DataSet is {}'.format(dataset.name))
  dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=int(opt.nThreads),
    collate_fn=collateClass)

  dataset_size = len(dataloader)
  print('#training images (CT) = %d' % dataset_size)

  # get Xray only data loader
  dataset_xray = AlignDataSet_SingleImage(opt)
    
  dataloader_xray = torch.utils.data.DataLoader(
    dataset_xray,
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=int(opt.nThreads),
    collate_fn=collateClass)
  print('#training images (X) = %d' % len(dataloader_xray))


  print("# batch size ", opt.batch_size)
    
  # get model
  gan_model = get_model(opt.model_class)()
  print('Model --{}-- will be Used'.format(gan_model.name))
  gan_model.init_process(opt)
  total_steps, epoch_count = gan_model.setup(opt)
        
  # load pretrained model  
  total_steps, epoch_count = gan_model.load_networks(opt.check_point, opt.load_path, opt.latest)
  
  # set to train
  gan_model.train()

  # visualizer
  from lib.utils.visualizer import Visualizer
  visualizer = Visualizer(log_dir=os.path.join(gan_model.save_root, 'train_log'))

  total_steps = total_steps

  # train discriminator more
  dataloader_iter_for_discriminator = iter(dataloader)

  # train    
  for epoch in range(opt.epoch_count):
    epoch_start_time = time.time()
    iter_data_time = time.time()

    # train with Xrays
    print('Training with XRay Images Only')
    nsteps = len(dataloader_xray)
    for epoch_i, data in enumerate(dataloader_xray):
      iter_start_time = time.time()

      total_steps += 1
      gan_model.set_input(data)
      t0 = time.time()
      gan_model.optimize_G_proj_norm()
      t1 = time.time()
      # loss
      loss_dict = gan_model.get_current_losses()
      # visualizer.add_scalars('Train_Loss', loss_dict, step=total_steps)
      total_loss = visualizer.add_total_scalar('Total loss', loss_dict, step=total_steps)

      #if total_steps % opt.print_freq == 0:
      print('X total step {}/{} : {} timer: {:.4f} sec.'.format(epoch_i , nsteps, total_steps, t1 - t0))
      #print('epoch {}, step{}:{} || total loss:{:.4f}'.format(epoch,
      #                                                             epoch_i, dataset_size, total_loss))
      #print('||'.join(['{}: {:.4f}'.format(k, v) for k, v in loss_dict.items()]))
      #print('')
   
    # train with CT
    print('Training with CT Images')
    for epoch_i, data in enumerate(dataloader):
      iter_start_time = time.time()

      total_steps += 1
      gan_model.set_input(data)
      t0 = time.time()
      gan_model.optimize_parameters()
      t1 = time.time()

        

      # loss
      loss_dict = gan_model.get_current_losses()
      # visualizer.add_scalars('Train_Loss', loss_dict, step=total_steps)
      total_loss = visualizer.add_total_scalar('Total loss', loss_dict, step=total_steps)

      #if total_steps % opt.print_freq == 0:
      print('CT total step: {} timer: {:.4f} sec.'.format(total_steps, t1 - t0))
      print('epoch {}, step{}:{} || total loss:{:.4f}'.format(epoch,
                                                               epoch_i, dataset_size, total_loss))
      print('||'.join(['{}: {:.4f}'.format(k, v) for k, v in loss_dict.items()]))
      print('')

    # save model several epoch
    #if epoch % opt.save_epoch_freq == 0 and epoch >= opt.begin_save_epoch:
    print('saving the model at the end of epoch %d, iters %d' %
            (epoch, total_steps))
    gan_model.save_networks(epoch, total_steps)

    print('End of epoch %d \t Time Taken: %d sec' %
          (epoch, time.time() - epoch_start_time))
    
    gan_model.update_learning_rate(epoch) # start from pretrained model
