#!/usr/bin/env python
# coding: utf-8

# In[3]:


import sys
sys.path.append('/scratch/es5223/medsynth/X2CT/3DGAN/')

DATAPATH_SOURCE = '/scratch/es5223/medsynth/data/TBX11K_classification_splits/'
OUT_SOURCE = '/scratch/es5223/medsynth/data_syntheticCT/TBX11K_orig_XCT_classification_splits/'


SUFFIX = sys.argv[1]

print(SUFFIX)

from lib.config.config import cfg_from_yaml, cfg, merge_dict_and_yaml, print_easy_dict
from lib.dataset.factory import get_dataset
from lib.model.factory import get_model
from lib.utils import html
from lib.utils.visualizer import tensor_back_to_unnormalization, save_images, tensor_back_to_unMinMax
#from lib.utils.metrics_np import MAE, MSE, Peak_Signal_to_Noise_Rate, Structural_Similarity, Cosine_Similarity
from lib.utils import ct as CT
import copy
import torch
import numpy as np
import os


import matplotlib.pyplot as plt

import glob
from collections import Counter
import cv2
from tqdm import tqdm


# ## Load Model and Evaluate

# In[4]:


# HD = 128
HD = 256
# HD = 512


# In[5]:


args = {'dataset_class':'align_ct_xray_std',
        'datasetfile':'data/test_small1.txt',
        'dataroot':'/LIDC-HDF5-256/',
        'model_class':'SingleViewCTGAN',
        'gpu_ids':[0],
        'data': 'LIDC256',
        'tag': 'd2_singleview2500',
        'check_point': 9,
        'how_many': 3,
        'load_path':None,
        'latest':False,
        'verbose':True,
        'save_path':'NONE',
        'imageGAN':False}
ymlpath = '/scratch/es5223/medsynth/X2CT/3DGAN/experiment/singleview2500/d2_singleview2500.yml'

if ymlpath is not None:
    cfg_from_yaml(ymlpath)
# merge config with argparse
opt = copy.deepcopy(cfg)
opt = merge_dict_and_yaml(args, opt)
print_easy_dict(opt)

opt.serial_batches = True

opt.model_save_directory = '/scratch/es5223/medsynth/X2CT/3DGAN/projection_model_experiments/1_norm/singleView_CTGAN/'
opt.model_save_directory += 'LIDC256/d2_singleview2500/checkpoint/'


print(ymlpath)


# In[6]:


# get model
gan_model = get_model(opt.model_class)()
print('Model --{}-- will be Used'.format(gan_model.name))

# set to test
gan_model.eval()

gan_model.init_process(opt)
total_steps, epoch_count = gan_model.setup(opt)


if 'batch' in opt.norm_G:
  gan_model.eval()
elif 'instance' in opt.norm_G:
  gan_model.eval()
  # instance norm in training mode is better
  for name, m in gan_model.named_modules():
    if m.__class__.__name__.startswith('InstanceNorm'):
      m.train()
else:
  raise NotImplementedError()


# ## check data loader/evaluate on other data

# In[7]:


from lib.dataset.baseDataSet import Base_DataSet
from lib.dataset.utils import *
import h5py
import numpy as np
import cv2

class AlignDataSet_SingleImage(Base_DataSet):
  '''
  DataSet For unaligned data
  '''
  def __init__(self, opt):
    super(AlignDataSet_SingleImage, self).__init__()
    self.opt = opt
    self.data_augmentation = self.opt.data_augmentation(opt)

  @property
  def name(self):
    return 'AlignDataSet'

  @property
  def get_data_path(self):
    path = os.path.join(self.opt.dataroot)
    return path

  @property
  def num_samples(self):
    return 1 # self.dataset_size

  '''
  generate batch
  '''
  def pull_item(self, item):
    ct_data = np.zeros((256,256,256))
    #sel = ['/scratch/es5223/medsynth/data/padchest_part/padchest_images/tb_test_images/216840111366964013076187734852011266095822272_00-135-158.png', 
    #       '/scratch/es5223/medsynth/data/padchest_part/padchest_images/tb_test_images/216840111366964013076187734852011266095822272_00-135-113.png']
    print(opt.selected_images)
    x_ray1_0 = cv2.imread(opt.selected_images[0], cv2.IMREAD_GRAYSCALE)    
    x_ray1_resized = cv2.resize(x_ray1_0, (256,256), interpolation = cv2.INTER_CUBIC)
    x_ray1 = np.expand_dims(x_ray1_resized,0)

    # print("x_ray1 ", x_ray1.shape)

#     file_path = '/scratch/es5223/medsynth/data/padchest_part/padchest_images/tb_test_images/'
    file_path = None
    # Data Augmentation
    ct, xray1 = self.data_augmentation([ct_data, x_ray1])

    return ct, xray1, file_path


# In[25]:
DATA_PATH = DATAPATH_SOURCE + SUFFIX

FOLDER_OUT = OUT_SOURCE + SUFFIX
os.makedirs(FOLDER_OUT, exist_ok=True)

names = glob.glob(DATA_PATH+'*')
all_pairs = [[name] for name in names] # subset

print(len(names))
# print(all_pairs[0:3])

for ind_i in tqdm(range(len(all_pairs))):
    # get model predictions
    opt.selected_images = all_pairs[ind_i]
    if os.path.isfile(FOLDER_OUT + 'CT_' + opt.selected_images[0].split('/')[-1].replace('.png','.npy')):
        continue

    datasetClass, _, dataTestClass, collateClass = get_dataset(opt.dataset_class)
    opt.data_augmentation = dataTestClass

    dataset = AlignDataSet_SingleImage(opt)
    dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=int(opt.nThreads),
    collate_fn=collateClass)

    dataset_size = len(dataloader)
    data = next(iter(dataloader))
    gan_model.set_input(data)
    gan_model.test()

    visuals = gan_model.get_current_visuals()
    ct_pred = visuals['G_fake'].cpu().squeeze().numpy()

    # save
    os.makedirs(FOLDER_OUT, exist_ok=True)
    # tmp = cv2.imwrite(FOLDER_OUT + 'CT_' + opt.selected_images[0].split('/')[-1], im_all)
    np.save(FOLDER_OUT + 'CT_' + opt.selected_images[0].split('/')[-1].replace('.png','.npy'), ct_pred)

