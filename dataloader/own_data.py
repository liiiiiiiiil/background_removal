#!/usr/bin/env python
import os
import collections
import os.path as osp
import numpy as np
import PIL.Image
import scipy.io
import torch
from torch.utils import data


class Own_data(data.Dataset):

    def __init__(self,root,transform=None):
        self.root = root
        self._transform = transform

        self.files=[]
        img_names=os.listdir(os.path.join(root,'images'))
        for img_name in img_names:
            img_file=os.path.join(root,'images/%s'%img_name)
            self.files.append(img_file)
    def __len__(self):
        return len(self.files)

    def __getitem__(self,index):
        img_file=self.files[index]

        #the output should have a name
        img_name=os.path.split(img_file)[-1] 
        img=PIL.Image.open(img_file)
        if self._transform!= None:
            return img_name,self._transform.transform(img)
        else:
            return img_name,img



