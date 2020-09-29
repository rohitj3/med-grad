# -*- coding: utf-8 -*-

import os, sys, pdb
import torch
from torchvision import transforms
import random
from .knee_sets import ImageFolder


def data_load(args):
    pixel_mean, pixel_std = 0.66133188,  0.21229856
    phases = ['train', 'val', 'test', 'auto_test']
    #phases = ['train','val']
    # phases = ['train', 'val', 'test', 'auto_test']
    data_transform = {
        'train': transforms.Compose([
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
            transforms.ToTensor(),
            transforms.Normalize([pixel_mean]*3, [pixel_std]*3)
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([pixel_mean]*3, [pixel_std]*3)
        ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([pixel_mean]*3, [pixel_std]*3)
        ]),
        'auto_test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([pixel_mean]*3, [pixel_std]*3)
        ]),
        'most_test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([pixel_mean]*3, [pixel_std]*3)
        ]),
        'most_auto_test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([pixel_mean]*3, [pixel_std]*3)
        ])
    }

                      
    
    
    dsets = {x: ImageFolder(os.path.join(args.data_dir, x), data_transform[x]) for x in phases}
    
    #dsets_lis = {x:sample(len(dsets[x])) for x in phases}
    #dsets_temp = {subset(dsets[x],dsets_lis[x]) for x in phases}
    print(len(dsets['train']))
    dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=args.batch_size,
            sampler=torch.utils.data.SubsetRandomSampler(list(range(len(dsets[x]))), generator=None)) for x in phases}
    dset_classes = dsets['train'].classes
    #print(dset_classes)
    dset_size = {x: len(dsets[x]) for x in phases}
    num_class = len(dset_classes)
    
    print(len(dset_loaders['train']))
    return dset_loaders, dset_size, num_class


def make_weights_for_balanced_classes(images, nclasses):                        
    count = [0] * nclasses                                                      
    for item in images:                                                         
        count[item[1]] += 1                                                     
    weight_per_class = [0.] * nclasses                                      
    N = float(sum(count))                                                   
    for i in range(nclasses):                                                   
        weight_per_class[i] = N/float(count[i])                                 
    weight = [0] * len(images)                                              
    for idx, val in enumerate(images):                                          
        weight[idx] = weight_per_class[val[1]]                                  
    return weight   

def sample(length):
    
    lis = []
    for i in range(length/2):
        lis.append(random.randrange(length))
    return lis
    