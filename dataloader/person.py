#!/usr/bin/env python

import collections
import os.path as osp
import numpy as np
import PIL.Image
import scipy.io
import torch
from torch.utils import data


class Person(data.Dataset):

    def __init__(self,root,split='train',transform=None):
        self.root = root
        self.split = split
        self.transform = transform

        self.files=collections.defaultdict(list)
        for split in ['train','val']:
            imgsets_file=osp.join(root,'%s.txt'%split)

            for did in open(imgsets_file):
                did=did.strip()
                img_file=osp.join(root,'clean_images/images/%s.jpg'%did)
                lbl_file=osp.join(root,'clean_images/profiles/%s.jpg'%did)
                self.files[split].append({
                    'img':img_file,
                    'lbl':lbl_file,
                    })
    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self,index):
        data_file=self.files[self.split][index]
        #load img
        img_file=data_file['img']
        img=PIL.Image.open(img_file)
        #load lbl
        lbl_file=data_file['lbl']
        lbl=PIL.Image.open(lbl_file)
        if transform!= None:
            return self.transform(img,lbl)
        else:
            return img,lbl



