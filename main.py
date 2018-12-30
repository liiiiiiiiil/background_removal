# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim 
from torch.optim import lr_scheduler 
from torch.utils.data import DataLoader 
from models.fcn import VGGNet,FCN32s,FCN16s,FCN8s,FCNs 

from dataloader.voc import SBDClassSeg
from dataloader.person import Person
from dataloader.own_data import Own_data

from train import train_epoch
from val import val_epoch,eval_own_images
from misc.utils import Cross_Entropy2D,cul_acc
from misc.transform import Trans

import numpy as np
import time
import os
import sys
import argparse


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--id',type=str,default=None,
            help='An id to distinguish the saved model')
    parser.add_argument('--use_cuda',type=int,default=True)
    parser.add_argument('--start_from',type=str,default=None)
    # parser.add_argument('--data_type',type=str,default='person')
    parser.add_argument('--data_type',type=str,default='own_images')
    # parser.add_argument('--data_root',type=str,default='/mnt/disk1/han/dataset/')
    # parser.add_argument('--data_root',type=str,default='/mnt/disk1/lihao/person_br/datasets/icome_task2_data')
    parser.add_argument('--data_root',type=str,default='/mnt/disk1/lihao/person_br/datasets/own_images/')
    parser.add_argument('--optimizer',type=str,default='rmsprop',
            help='Choose a optimizer')
    parser.add_argument('--max_epochs',type=int,default=40)
    parser.add_argument('--max_val_iterations',type=int,default=100,
            help='When you don\'t want to test the full val datasets,this help a lot')
    parser.add_argument('--batch_size',type=int,default=16)
    parser.add_argument('--n_class',type=int,default=2)
    parser.add_argument('--lr',type=float,default=5e-4)
    parser.add_argument('--momentum',type=float,default=0)
    parser.add_argument('--w_decay',type=float,default=1e-5)
    parser.add_argument('--update_lr_rate',type=float,default=0.8)
    parser.add_argument('--update_lr_step',type=float,default=5)
    
    parser.add_argument('--checkpoint_every',type=int,default=10)
    parser.add_argument('--checkpoint_save_path',type=str,default='/mnt/disk1/lihao/person_br/save/')
    parser.add_argument('--image_save_path',type=str,default='/mnt/disk1/lihao/person_br/save/imgs4/',
            help='Where save the images')

    args=parser.parse_args()

    kwargs={"num_workers":4,"pin_memory":True} if args.use_cuda else {}
    
    if args.data_type=='voc':
        train_loader=DataLoader(SBDClassSeg(args.data_root,
            split='train',transform=True),batch_size=args.batch_size,
            shuffle=True,**kwargs)
        val_loader=DataLoader(SBDClassSeg(args.data_root,
            split='val',transform=True),batch_size=args.batch_size,
            shuffle=False,**kwargs)
    elif args.data_type=='person':
        mean_bgr=np.array([128.0523,134.4394,141.8439])
        transform=Trans(512,384,mean_bgr)
        train_loader=DataLoader(Person(args.data_root,
            split='train',transform=transform),batch_size=args.batch_size,
            shuffle=True,**kwargs)
        val_loader=DataLoader(Person(args.data_root,
            split='val',transform=transform),batch_size=args.batch_size,
            shuffle=True,**kwargs)
    elif args.data_type=='own_images':
        mean_bgr=np.array([128.0523,134.4394,141.8439])
        transform=Trans(512,384,mean_bgr)
        data_loader=DataLoader(Own_data(args.data_root,
            transform=transform),batch_size=args.batch_size,
            shuffle=False,**kwargs)
    else:
        raise RuntimeError('No this dataset')
   
    vgg_model=VGGNet(requires_grad=True,remove_fc=True)
    model=FCNs(pretrained_net=vgg_model,n_class=args.n_class)
    if args.use_cuda:
        vgg_model=vgg_model.cuda()
        model=model.cuda()
        model=nn.DataParallel(model)
    
    criterion=Cross_Entropy2D()
    if args.optimizer=='rmsprop':
        optimizer=optim.RMSprop(model.parameters(),lr=args.lr,
            momentum=args.momentum,weight_decay=args.w_decay)
    elif args.optimizer=='adam':
        optimizer=optim.Adam(model.parameters(),lr=args.lr,
                weight_decay=args.w_decay)
    else:
        print('Please use adam or rmsprop as your optimizer')
        raise RuntimeError('Wrong optimizer')
        
    scheduler=lr_scheduler.StepLR(optimizer,step_size=args.update_lr_step,gamma=args.update_lr_rate)


    infos={}
    infos['iteration']=0
    infos['epoch']=0
    infos['train_loss']=[]
    infos['val_loss']=[]
    infos['mean_Pixel']=[]
    infos['meanIU']=[]

    if args.start_from is not None and os.path.isfile(args.start_from):
        D=torch.load(args.start_from)
        if '_best_' in args.start_from:
            infos=D['infos']
            model.load_state_dict(D['model_state_dict'])
        else:
            infos=D['infos']
            optimizer.load_state_dict(D['optimizer_state_dict'])
            scheduler.load_state_dict(D['scheduler_state_dict'])
            model.load_state_dict(D['model_state_dict'])

    epoch=infos['epoch']
    if args.data_type=='own_images':
        eval_own_images(model,data_loader,args)
        return 

    best_acc=-1
    for i in range(epoch,args.max_epochs):
        #Train epoch
        train_epoch(model,optimizer,criterion,train_loader,infos,args)
        #Val epoch
        val_epoch(model,criterion,val_loader,infos,args)
        #Save infos and model_dict
        scheduler.step()
        
        acc=cul_acc(infos['mean_Pixel'][-1][1],infos['meanIU'][-1][1])
        if acc>best_acc:
            torch.save({
                'infos':infos,
                'model_state_dict':model.state_dict(),
                },os.path.join(args.checkpoint_save_path,'model_best_'+args.id+'.pkl'))
            best_acc=acc
        torch.save({
            'infos':infos,
            'model_state_dict':model.state_dict(),
            'optimizer_state_dict':optimizer.state_dict(),
            'scheduler_state_dict':scheduler.state_dict(),
            },os.path.join(args.checkpoint_save_path,'model_'+args.id+'.pkl'))
        infos['epoch']+=1

if __name__=='__main__':
    main()
        
        

    



    

    

    

