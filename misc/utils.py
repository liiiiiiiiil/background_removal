import torch 
import torch.nn as nn
import numpy as np
import os
import PIL.Image as Image
import torch.nn.functional as F
from distutils.version import LooseVersion

class Cross_Entropy2D(nn.Module):
    def __init__(self,weight=None,size_average=True):
        super().__init__()
        self.weight=weight
        self.size_average=size_average

    def forward(self,input,target):#input:n,c,h,w target:n,h,w
        n, c, h, w = input.size()
        # log_p: (n, c, h, w)
        if LooseVersion(torch.__version__) < LooseVersion('0.3'):
            # ==0.2.X
            log_p = F.log_softmax(input)
        else:
        # >=0.3
            log_p = F.log_softmax(input, dim=1)
        # log_p: (n*h*w, c)
        log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
        log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
        log_p = log_p.view(-1, c)
        # target: (n*h*w,)
        mask = target >= 0
        target = target[mask]
        loss = F.nll_loss(log_p, target,reduction='sum')
        if self.size_average:
            mask=mask.float()
            loss /= mask.data.sum()
        return loss
        

def cross_entropy2d(input, target, weight=None, size_average=True):
    # input: (n, c, h, w), target: (n, h, w)
    n, c, h, w = input.size()
    # log_p: (n, c, h, w)
    if LooseVersion(torch.__version__) < LooseVersion('0.3'):
        # ==0.2.X
        log_p = F.log_softmax(input)
    else:
        # >=0.3
        log_p = F.log_softmax(input, dim=1)
    # log_p: (n*h*w, c)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)
    # target: (n*h*w,)
    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, weight=weight, reduction='sum')
    if size_average:
        loss /= mask.data.sum()
    return loss

def save2image(image_array,img_names,save_path):
    #image_array: batch_size,h,w

    image_array=image_array.astype(np.uint8)
    num=image_array.shape[0]  
    #Here,saving the input img and output img
    for i in range(num):
        img_o=Image.fromarray(image_array[i])
        img_name_o=img_names[i]
        img_o.save(os.path.join(save_path,img_name_o))

def iou(pred, target,n_class):
    ious = []
    for cls in range(n_class):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = pred_inds[target_inds].sum()
        union = pred_inds.sum() + target_inds.sum() - intersection
        if union == 0:
            ious.append(float('nan'))  # if there is no ground truth, do not include in evaluation
        else:
            ious.append(float(intersection) / max(union, 1))
        # print("cls", cls, pred_inds.sum(), target_inds.sum(), intersection, float(intersection) / max(union, 1))
    return ious


def pixel_acc(pred, target):
    correct = (pred == target).sum()
    total   = (target == target).sum()
    return correct / total


