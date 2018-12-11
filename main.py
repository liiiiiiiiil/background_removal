# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim 
from torch.optim import lr_scheduler 
from torch.utils.data import DataLoader 
from fcn import VGGNet,FCN32s,FCN16s,FCN8s,FCNs 
from datasets.voc import SBDClassSeg
from train import train_epoch

import numpy as np
import time
import os
import sys
import argparse


def main():
    parser=argparse.ArgumentParser()
    parser=add_argument('--id',type=str,default=None,
            help='An id to distinguish the saved model')
    parser.add_argument('--use_cuda',type=int,default=True)
    parser.add_argument('--start_from',type=str,default=None)
    parser.add_argument('--data_type',type=str,default='voc')
    parser.add_argument('--data_root',type=str,default=None)

    parser.add_argument('--max_epochs',type=int,default=20)
    parser.add_argument('--batch_size',type=int,default=1)
    parser.add_argument('--n_class',type=int,default=20)
    parser.add_argument('--lr',type=float,default=1e-4)
    parser.add_argument('--momentum',type=float,default=0)
    parser.add_argument('--w_decay',type=float,default=1e-5)
    parser.add_argument('--update_lr_rate',type=float,default=0.5)
    parser.add_argument('--update_lr_step',type=float,default=500)
    
    parser.add_argument('--checkpoint_every',type=int,default=100)
    parser.add_argument('--save_path',type=str,default=None)
    parser.add_argument('--num_images_save',type=int,default=5,
            help='How much images you want to save')
    parser.add_argument('--image_save_path',type=int,default=None,
            help='Where save the images')

    args=parser.parse_args()

    kwargs={"num_workers":4,"pin_memory":True} if args.use_cuda else {}
    
    if args.data_type=='voc':
        train_loader=DataLoader(SBDClassSeg(args.data_root,
            split='train',transform=True),batch_size=batch_size,
            shuffle=True,**kwargs)
        val_loader=DataLoader(SBDClassSeg(args.data_root,
            split='val',transform=True),batch_size=batch_size,
            shuffle=False,**kwargs)

    
    vgg_model=VGGNet(requires_grad=True,remove_fc=True)
    model=FCNs(pretrained_net=vgg_model,n_class=n_class)
    if args.use_cuda:
        vgg_model=vgg_model.cuda()
        model=model.cuda()
        model=nn.DataParallel(model)
    
    criterion=nn.BCEWithLogitsLoss()
    optimizer=optim.RMSprop(model.parameters(),lr=lr,
            momentum=momentum,weight_decay=w_decay)
    if args.use_cuda:
        optimizer=optimizer.cuda()
    scheduler=lr_scheduler.StepLR(optimizer,step_size=update_lr_step,gamma=update_lr_rate)


    infos={}
    infos['iteration']=0
    infos['epoch']=0
    infos['train_loss']=[]
    infos['val_loss']=[]
    infos['mean_Pixel']=[]
    infos['meanIU']=[]

    if start_from is not None and os.path.isfile(os.path.join(args.start_from,'model_'+args.id+'.pkl')):
        D=torch.load(os.path.join(args.start_from,'model_'+args.id+'.pkl'))
        infos=D['infos']
        model.load_state_dict(D['model_state_dict'])
        optimizer.load_state_dict(D['optimizer_state_dict'])
        scheduler.load_state_dict(D['scheduler_state_dict'])


    for epoch in range(infos['epoch'],args.max_epochs):
        #train epoch
        train_epochs(model,optimizer,scheduler,criterion,train_loader,infos,args)
        #val epoch
        val_epoch(model,criterion,val_loader,infos,args)
        #save infos and model_dict
        infos['epoch']=epoch
        torch.save({
            'infos':infos,
            'model_state_dict':model.state_dict(),
            'optimizer_state_dict':optimizer.state_dict(),
            'scheduler_state_dict':scheduler.state_dict()
            },os.path.join(args.save_path,'model_'+args.id+'.pkl'))

        
        

    



    

    

    

