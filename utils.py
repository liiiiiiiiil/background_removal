import torch 
import numpy as np
import os
import PIL.Image as Image

def save2image(image_array,args):
    #image_array: batch_size,h,w

    N=image_array.shape[0]
    N_=args.num_images_save
    if N<N_:
        image_array=image_array
    else:
        image_array=image_array[0:N_]
    
    num=image_array.shape[0]  
    for i in range(num):
        img=Image.fromarray(image_array[i])
        image_name=str(i)+'.png'
        img.save(os.path.join(args.image_save_path,image_name))

def iou(pred, target):
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


