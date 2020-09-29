# -*- coding: utf-8 -*-

import os, sys, pdb
import shutil
import torch
from torch import nn, optim
from torch.autograd import Variable
import time

import utils
from utils.torch_util import LRScheduler
from utils.loss_util import weighted_loss
from utils.eval_eng import eval_test

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def train_model(args, model, dset_loaders, dset_size):
    best_model, best_acc, best_num_epoch = None, .0, 0
    best_model_path = os.path.join(args.model_dir, args.best_model_name, str(args.session))
    if os.path.exists(best_model_path):
        shutil.rmtree(best_model_path)
    os.makedirs(best_model_path)
    
    criterion = nn.CrossEntropyLoss()
    if args.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr,
                              momentum=0.9, weight_decay=args.weight_decay)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr,
                               weight_decay=args.weight_decay)
    elif args.optim == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay, momentum=0.9)

    # print('-'*10+'Training'+'-'*10)
    print('Initial lr:{}  Optimizer:{}  network:{}  depth:{}  num_class:{}'.format(
        args.lr, args.optim, args.net_type, args.depth, args.num_class))

    lr_scheduler = LRScheduler(args.lr, args.lr_decay_epoch)
    for epoch in range(args.num_epoch):
        since = time.time()
        print('Epoch {}/{}'.format(epoch, args.num_epoch))
        for phase in ['train', 'val']:
            if phase == 'train':
                optimizer = lr_scheduler(optimizer, epoch)
                model.train(True)
            else:
                model.train(False)

            running_loss, running_corrects = .0, .0

            for data in dset_loaders[phase]:
                inputs, labels, _ = data
                #Modified
                #inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                #inputs, labels = Variable(inputs.cpu()), Variable(labels.cpu())
                
                #inputs = inputs.to(device)
                #labels = labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                #outputs = outputs.to(device)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                _, preds = torch.max(outputs.data, 1)#.to(device)
                if args.wloss == False:
                    loss = criterion(outputs, labels)#.to(device)
                else:
                    loss = weighted_loss(outputs, labels, args)#.to(device)

                if phase == 'train':
                    print("Calculating Loss")
                    loss.backward()
                    print("Optimizing Loss")
                    optimizer.step()
                print("Running Loss Calculation")
                running_loss += loss.data
                running_corrects += torch.sum(preds == labels.data)
                
            #epoch_loss = 1.0*running_loss.cpu().tolist()/dset_size[phase]
            epoch_loss = 1.0*running_loss.tolist()/dset_size[phase]
            epoch_acc = 1.0*running_corrects.tolist()/dset_size[phase]
            elapse_time = time.time() - since
            
            print("In {}, Number case:{} Loss:{:.4f} Acc:{:.4f} Time:{}".format(
                 phase, dset_size[phase], epoch_loss, epoch_acc, elapse_time))

            if phase == "val" and epoch_acc >= best_acc:
                best_acc = epoch_acc
                best_num_epoch = epoch
                val_metric_str = str(epoch).zfill(2) + '-' + str(round(best_acc, 3))
                test_acc, test_mse = eval_test(args, model, dset_loaders, dset_size, "test")
                test_metric_str = "-" + str(round(test_acc, 3)) + "-" + str(round(test_mse, 3)) + ".pth"
                args.best_model_path = os.path.join(best_model_path, val_metric_str + test_metric_str)
                torch.save(model, args.best_model_path)
                #torch.save(model.cpu(), args.best_model_path)
                print("---On test_set: acc is {:.3f}, mse is {:.3f}".format(test_acc, test_mse))
                #model.cuda()

    print('='*80)
    print ('Validation best_acc: {}  best_num_epoch: {}'.format(best_acc, best_num_epoch))
