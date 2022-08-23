import os
import re
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


for TRIAL in range(2,3):
    CHECK_POINT_PATH = 'models_trials/3D_CNN_bce+'+str(TYPE)+str(TRIAL)+'.pth'

    torch.backends.cudnn.benchmark = True

    def load_image(path):
        ct_array = np.load(path)
        # image = Image.fromarray(ct_array[63,:,:])#.convert('RGB')
        image = ct_array[32:96,:,:].transpose(2, 1, 0) #.convert('RGB')
        return image

    def is_valid_file(path):
        return True

    device = torch.device("cuda:0")
    batch_size = 8 #80
    num_epochs = 500
    lr = 0.01
    beta = 1
    step_size = 130
    # img_size = 224
    img_size = 128

    test_size = int((256 / 224) * img_size)

    mean = [0.485] 
    std = [0.229]
    num_workers = 0

    data_transforms_ct = {
        'train': transforms.Compose([
            #transforms.RandomRotation(30),
            #transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(0., 1.),
            #transforms.RandomErasing(),
        ]),
        'val': transforms.Compose([
            #transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(0., 1.),
        ]),
        'test': transforms.Compose([
            #transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(0., 1.),
        ])
    }


    # Load the datasets with ImageFolder
    print("data_dir ", data_dir)
    # image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val', 'test']}

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), 
                                              data_transforms_ct[x],
                                              loader=load_image,
                                              is_valid_file=is_valid_file) for x in ['train', 'val', 'test']}

    class_names = image_datasets['train'].classes
    num_classes = len(class_names)
    data_loader = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                 shuffle=True, num_workers=num_workers, pin_memory = True)
                  for x in ['train', 'val', 'test']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    NUM_CLASSES = len(image_datasets['train'].classes)

    print(class_names)
    print(dataset_sizes)
    print(device)

    ### we get the class_to_index in the data_Set but what we really need is the cat_to_names  so we will create
    _ = image_datasets['val'].class_to_idx
    cat_to_name = {_[i]: i for i in list(_.keys())}
    print(cat_to_name)

    # Run this to test the data loader
    images, labels = next(iter(data_loader['val']))
    print("images.size() ", images.size())

    class CNN3D(torch.nn.Module):
        def __init__(self):
            super(CNN3D, self).__init__()
            """Build a 3D convolutional neural network model."""
            self.relu = torch.nn.ReLU()
            self.maxPool = torch.nn.MaxPool3d(kernel_size=2)


            self.conv1 = torch.nn.Conv3d(1, 64, kernel_size=(3,3,3))
            self.batchNorm1 = torch.nn.BatchNorm3d(num_features=64)


            self.conv2 = torch.nn.Conv3d(64, 64, kernel_size=(3,3,3))
            self.batchNorm2 = torch.nn.BatchNorm3d(num_features=64)

            self.conv3 = torch.nn.Conv3d(64, 128, kernel_size=(3,3,3))
            self.batchNorm3 = torch.nn.BatchNorm3d(num_features=128)

            self.conv4 = torch.nn.Conv3d(128, 256, kernel_size=(3,3,3))
            self.batchNorm4 = torch.nn.BatchNorm3d(num_features=256)


            self.avgPool = torch.nn.AvgPool3d(kernel_size=(2,6,6))
            self.fc1 = torch.nn.Linear(256, 512)
            self.dropout = torch.nn.Dropout3d(0.3)
            #self.fc2 = torch.nn.Linear(512, 1)
            self.sigmoid = torch.nn.Sigmoid()
            self.fc2 = torch.nn.Linear(512, NUM_CLASSES)

        def forward(self, x):
            x = self.conv1(x)
            x = self.relu(x)
            x = self.maxPool(x)
            x = self.batchNorm1(x)

            x = self.conv2(x)
            x = self.relu(x)
            x = self.maxPool(x)
            x = self.batchNorm2(x)

            x = self.conv3(x)
            x = self.relu(x)
            x = self.maxPool(x)
            x = self.batchNorm3(x)

            x = self.conv4(x)
            x = self.relu(x)
            x = self.maxPool(x)
            x = self.batchNorm4(x)

            x = self.avgPool(x).squeeze()
            x = self.fc1(x)

            x = self.dropout(x)
            x = self.fc2(x)
            x = self.sigmoid(x)#.squeeze()
            return x

    model = CNN3D()
    model

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001) # todo same schedule
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96, last_epoch=- 1, verbose=False)

    model.to(device)

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
                    inputs = inputs.to(device)
                    labels_orig = labels.cuda()
                    labels = torch.nn.functional.one_hot(labels, num_classes=NUM_CLASSES)

                    labels = labels.to(device).float()
                    inputs = inputs.unsqueeze(1)
                    r = np.random.rand(1)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    #if i % 1000 == 999:
                    #if i % 10 == 9:
                    if i>0:
                        print('[%d, %d/%d] loss: %.8f' % 
                              (epoch + 1, i,len(data_loader[phase]), running_loss / (i * inputs.size(0))))

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                        _, preds = torch.max(outputs, 1)
                        if phase == 'train':                
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels_orig.data)

                if phase == 'train':                
                    scheduler.step()

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
                    best_epoch = epoch
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save({'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'scheduler_state_dict': scheduler.state_dict(),
                                'best_val_loss': best_loss,
                                'best_val_accuracy': best_acc
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
        print('Best epoch ', best_epoch) 

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model, best_loss, best_acc

    try:
        checkpoint = torch.load(CHECK_POINT_PATH)
        print("checkpoint loaded")
    except:
        checkpoint = None
        print("checkpoint not found")
    if checkpoint == None:
        CHECK_POINT_PATH = CHECK_POINT_PATH

    if TRAIN:
        num_epochs = 50
        model, best_val_loss, best_val_acc = train_model(model,
                                                         criterion,
                                                         optimizer,
                                                         scheduler=scheduler,
                                                         num_epochs = num_epochs,
                                                         checkpoint = checkpoint
                                                         ) 


        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'best_val_accuracy': best_val_acc,
                    }, CHECK_POINT_PATH)

    def compute_validate_meter_bce(model, val_loader): # best_model_path,

        since = time.time()
        try:
            checkpoint = torch.load(CHECK_POINT_PATH)
            print("checkpoint loaded")
        except:
            checkpoint = None
            print("checkpoint not found")

        def load_model(best_model_path):                                
            model.load_state_dict(checkpoint['model_state_dict'])
            best_model_wts = copy.deepcopy(model.state_dict())
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            best_loss = checkpoint['best_val_loss']
            best_acc = checkpoint['best_val_accuracy']
        load_model(CHECK_POINT_PATH)
        model.to(device)
        model.eval()
        pred_y = list()
        test_y = list()
        probas_y = list()
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                target_orig = target
                target = torch.nn.functional.one_hot(target, num_classes=NUM_CLASSES)
                target = target.float()
                data = data.unsqueeze(1)

                data, target = Variable(data), Variable(target)
                output = model(data)
                probas_y.extend(output.data.cpu().numpy().tolist())
                pred_y.extend(output.data.cpu().max(1, keepdim=True)[1].numpy().flatten().tolist())
                test_y.extend(target_orig.data.cpu().numpy().flatten().tolist())

            # compute the confusion matrix
            confusion = confusion_matrix(test_y, pred_y)

            # plot the confusion matrix
            plot_labels = ['HEALTH', 'SICK', 'TB']
            #plot_confusion_matrix(confusion, plot_labels)
            #plot_confusion_matrix(confusion, classes=val_loader.dataset.classes,title='Confusion matrix')
            # print Recall, Precision, F1-score, Accuracy
            report = classification_report(test_y, pred_y, digits=4)
            print(report)        
        time_elapsed = time.time() - since

        print('Inference completes in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    count = count_parameters(model)
    print(count)

    compute_validate_meter_bce(model, data_loader['test']) #best_model_path,


