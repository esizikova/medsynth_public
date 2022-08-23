import os
import re
import cv2
import PIL
import sys
import json
import time
import timm
import math
import copy
import torch
import pickle
import logging
import fnmatch
import argparse
import itertools
import torchvision
from util import *
from model_cnn import *
import numpy as np
import pandas as pd
from apex import amp
import seaborn as sns
import albumentations
import torch.nn as nn
from PIL import Image
from pathlib import Path
from copy import deepcopy
import scikitplot as skplt
from sklearn import metrics
import torch.optim as optim
from timm import create_model
from datetime import datetime
from timm.data.loader import *
from torchvision import models
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.utils.data as data
from cutmix.cutmix import CutMix
from torch.autograd import Variable
from tqdm import tqdm, tqdm_notebook
from torch.optim import lr_scheduler
#from pytorch_metric_learning import loss
import torch.utils.model_zoo as model_zoo
from timm.models.layers.activations import *
from timm.utils import accuracy, AverageMeter
from apex.parallel import convert_syncbn_model
from timm.utils import ApexScaler, NativeScaler
from cutmix.utils import CutMixCrossEntropyLoss
from timm.models.registry import register_model
from collections import OrderedDict, defaultdict
from torch.utils.tensorboard import SummaryWriter
from warmup_scheduler import GradualWarmupScheduler
from timm.models.resnet import resnet26d, resnet50d
from torchvision import transforms, models, datasets
from timm.models.helpers import build_model_with_cfg
from torch.utils.data.sampler import SubsetRandomSampler
from randaugment import RandAugment, ImageNetPolicy, Cutout
from apex.parallel import DistributedDataParallel as ApexDDP
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.models.efficientnet import efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score, roc_curve, auc, roc_auc_score
#from timm.data import Dataset, DatasetTar, RealLabelsImagenet, create_loader, Mixup, FastCollateMixup, AugMixDataset

TRAIN = False
TYPE = sys.argv[1]

if TYPE=='orig':
    data_dir = '/scratch/es5223/medsynth/data_syntheticCT/TBX11K_orig_classification_splits/'
elif TYPE=='cyclegan':
    data_dir = '/scratch/es5223/medsynth/data_syntheticCT/TBX11K_cycleGAN_classification_splits/'
elif TYPE=='cyclegan+ns':
    data_dir = '/scratch/es5223/medsynth/data_syntheticCT/TBX11K_cycleGAN_n2s_classification_splits/'
elif TYPE=='cyclegan+nv':
    data_dir = '/scratch/es5223/medsynth/data_syntheticCT/TBX11K_cycleGAN_n2v_classification_splits/'
elif TYPE=='proj':
    data_dir = '/scratch/es5223/medsynth/data_syntheticCT/TBX11K_orig_XCTjoin_norm_finetune_classification_splits/'
elif TYPE=='proj+ns':
    data_dir = '/scratch/es5223/medsynth/data_syntheticCT/TBX11K_orig_XCTjoin_norm_finetune_n2s_classification_splits/'
elif TYPE=='proj+nv':
    data_dir = '/scratch/es5223/medsynth/data_syntheticCT/TBX11K_orig_XCTjoin_norm_finetune_n2v_classification_splits/'


    
for TRIAL in range(3):
    CHECK_POINT_PATH = 'models_trials/X+CT-joinedModel-3DCNN_CT'+str(TYPE)+'_TRIAL_'+str(TRIAL)+'.pth'
    
    torch.backends.cudnn.benchmark = True

    def load_image_joint(ct_path):
        ct_array = np.load(ct_path)
        tmp = ct_path.replace(data_dir,
                        '/scratch/es5223/medsynth/data/TBX11K_classification_splits/')
        tmp = tmp.replace('CT_','')
        tmp = tmp.replace('.npy','.png')
        if '.jpg.png' in tmp:
            tmp = tmp.replace('.png','')
        if not os.path.isfile(tmp):
            tmp = tmp.replace('.png','.jpg')
        if not os.path.isfile(tmp):
            print('ct_path ', ct_path)
            print('x_path ', tmp)

        x_img = cv2.imread(tmp, cv2.IMREAD_GRAYSCALE) * 1/256.
        x_image = cv2.resize(x_img,dsize=(224,224))
        # c_img = cv2.resize(ct_array[63,:,:],dsize=(224,224))
        # c_image = ct_array[32:96,:,:].transpose(2, 1, 0) #.convert('RGB')
        c_image = ct_array[:,:,:].transpose(2, 1, 0) #.convert('RGB')
        return [np.expand_dims(c_image, axis=0), np.expand_dims(x_image, axis=0)] #img_concat

    def is_valid_file(path):
        return True

    device = torch.device("cuda:0")
    batch_size = 10
    num_epochs = 500
    lr = 0.01
    beta = 1
    step_size = 130
    img_size_c = 128
    img_size = 224
    test_size = int((256 / 224) * img_size)

    mean = [0.485] 
    std = [0.229]
    num_workers = 0
    # Define your transforms for the training and testing sets
    data_transforms_x = {
        'train': transforms.Compose([
            #transforms.Resize(224),
            #transforms.ToTensor(),
            transforms.Normalize(0., 1.),
            transforms.RandomErasing()
        ]),
        'val': transforms.Compose([
            #transforms.Resize(224),
            #transforms.ToTensor(),
            transforms.Normalize(0., 1.)
            #transforms.RandomErasing()
        ]),
        'test': transforms.Compose([
            #transforms.Resize(224),
            #transforms.ToTensor(),
            transforms.Normalize(0., 1.)
            #transforms.RandomErasing()
        ])
    }


    data_transforms_ct = {
        'train': transforms.Compose([
            #transforms.RandomRotation(30),
            #transforms.Resize(img_size),
            #transforms.ToTensor(),
            transforms.Normalize(0., 1.),
            #transforms.RandomErasing(),
        ]),
        'val': transforms.Compose([
            #transforms.Resize(img_size),
            #transforms.ToTensor(),
            transforms.Normalize(0., 1.),
        ]),
        'test': transforms.Compose([
            #transforms.Resize(img_size),
            #transforms.ToTensor(),
            transforms.Normalize(0., 1.),
        ])
    }


    # Load the datasets with ImageFolder
    print("data_dir ", data_dir)

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), 
                                              #data_transforms_ct[x],
                                              loader=load_image_joint,
                                              is_valid_file=is_valid_file) for x in ['train', 'val', 'test']}

    class_names = image_datasets['train'].classes
    num_classes = len(class_names)
    data_loader = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                 shuffle=True, num_workers=num_workers, pin_memory = True)
                  for x in ['train', 'val', 'test']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}

    print(class_names)
    print(dataset_sizes)
    print(device)

    ### we get the class_to_index in the data_Set but what we really need is the cat_to_names  so we will create
    _ = image_datasets['val'].class_to_idx
    cat_to_name = {_[i]: i for i in list(_.keys())}
    print(cat_to_name)

    # Run this to test the data loader
    images, labels = next(iter(data_loader['val']))
    print("len(images) ", len(images))
    print("images[0].shape ", images[0].shape)
    
    # create model
    model = Two_Stream_Transformer(num_classes=3)
    print(model)
    
    # Create classifier
    for param in model.parameters():
        param.requires_grad = True

    criterion = LabelSmoothingCrossEntropy()
    #criterion = CutMixCrossEntropyLoss(True)
    #criterion = nn.CrossEntropyLoss()
    #optimizer = Nadam(model.parameters(), lr=0.001)
    #optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)


    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
    #lr = lambda x: (((1 + math.cos(x * math.pi / num_epochs)) / 2) ** 1) * 0.9
    #scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr)
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=5, after_scheduler=scheduler)
    #model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
    #loss_scaler = ApexScaler()
    #show our model architechture and send to GPU
    model.to(device)
    model.model_x.to(device)
    model.model_ct.to(device)


    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    count = count_parameters(model)
    print("The number of parameters of the model is:", count)


    def train_model(model, criterion, optimizer, scheduler, num_epochs=200, checkpoint = None):
        since = time.time()

        if checkpoint is None:
            best_model_wts = copy.deepcopy(model.state_dict())
            best_loss = math.inf
            best_acc = 0.
        else:
            print(f'Val loss: {checkpoint["best_val_loss"]}, Val accuracy: {checkpoint["best_val_accuracy"]}')
            model.load_state_dict(checkpoint['model_state_dict'])
            best_model_wts = copy.deepcopy(model.state_dict())
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            best_loss = checkpoint['best_val_loss']
            best_acc = checkpoint['best_val_accuracy']

        # Tensorboard summary
        writer = SummaryWriter()
        start_time_per_epoch = time.time()

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch + 1, num_epochs)) #(epoch, num_epochs -1)
            print('-' * 20)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for i, (inputs, labels) in enumerate(data_loader[phase]):
                    start_time = time.time()
                    labels = labels.to(device)

                    ct_data = inputs[0]
                    x_data = inputs[1]

                    ct_data_transformed = data_transforms_ct[phase](ct_data)
                    x_data_transformed = data_transforms_x[phase](x_data)

                    ct_data_transformed = ct_data_transformed.float().to(device)
                    x_data_transformed = x_data_transformed.float().to(device)

                    '''
                    r = np.random.rand(1)
                    if r < 0.5: #cutmix_prob=0.5
                    # generate mixed sample
                        lam = np.random.beta(beta, beta)
                        rand_index = torch.randperm(inputs.size()[0]).to(device)
                        target_a = labels
                        target_b = labels[rand_index]
                        bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
                        inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
                    # adjust lambda to exactly match pixel ratio
                        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))
                    ''' 
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    #if i % 1000 == 999:
                    #if i % 10 == 9:
                    total_time = time.time() - start_time


                    if i>0:
                        print('[%d, %d/%d] loss: %.8f time: %.3f' % 
                              (epoch + 1, i,len(data_loader[phase]), running_loss / (i * inputs[0].size(0)),total_time))

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(ct_data_transformed, x_data_transformed)

                        if False: #r < 0.5:
                            loss = criterion(outputs, target_a) * lam + criterion(outputs, target_b) * (1. - lam)
                        else:
                            loss = criterion(outputs, labels)

                        #loss = criterion(outputs, labels)
                        _, preds = torch.max(outputs, 1)

                        # backward + optimize only if in training phase
                        if phase == 'train':                
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs[0].size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':                
                    scheduler_warmup.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print('{} Loss: {:.8f} Acc: {:.8f}'.format(
                    phase, epoch_loss, epoch_acc))

                # Record training loss and accuracy for each phase
                if phase == 'train':
                    writer.add_scalar('Train/Loss', epoch_loss, epoch)
                    writer.add_scalar('Train/Accuracy', epoch_acc, epoch)
                    writer.flush()
                else:
                    writer.add_scalar('Valid/Loss', epoch_loss, epoch)
                    writer.add_scalar('Valid/Accuracy', epoch_acc, epoch)
                    writer.flush()
                # deep copy the model

                if phase == 'val' and epoch_acc > best_acc:
                    print(f'New best model found!')
                    print(f'New record ACC: {epoch_acc}, previous record acc: {best_acc}')
                    best_loss = epoch_loss
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save({'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'best_val_loss': best_loss,
                                'best_val_accuracy': best_acc,
                                'scheduler_state_dict' : scheduler.state_dict(),
                                }, 
                                CHECK_POINT_PATH
                                )
                    print(f'New record acc is SAVED: {epoch_acc}')

            end_time_per_epoch = (time.time() - start_time_per_epoch)
            print('Time for training the last epoch: {:.0f}m {:.0f}s'.format(
            end_time_per_epoch // 60, end_time_per_epoch % 60))

        time_elapsed = time.time() - since
        print('Total training time complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:.8f} Best val loss: {:.8f}'.format(best_acc, best_loss))

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model, best_loss, best_acc


    try:
        checkpoint = torch.load(CHECK_POINT_PATH)
        print("checkpoint loaded ", CHECK_POINT_PATH)
    except:
        checkpoint = None
        print("checkpoint not found ", CHECK_POINT_PATH)
    if checkpoint == None:
        CHECK_POINT_PATH = CHECK_POINT_PATH


    if TRAIN:
        num_epochs = 50
        model, best_val_loss, best_val_acc = train_model(model,
                                                         criterion,
                                                         optimizer,
                                                         scheduler,
                                                         num_epochs = num_epochs,
                                                         checkpoint = checkpoint
                                                         ) 


        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                    'best_val_accuracy': best_val_acc,
                    'scheduler_state_dict': scheduler.state_dict(),
                    }, CHECK_POINT_PATH)

    compute_validate_meter_ct_x_3dcnn(model, data_loader['test'], CHECK_POINT_PATH, data_transforms_ct,data_transforms_x) 


