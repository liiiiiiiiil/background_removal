from torchvision.transforms import Resize
import numpy as np
import PIL.Image as Image
import torch 


class Trans(object):

    def __init__(self,New_H,New_W,mean_bgr):

        self.resize=Resize([New_H,New_W])
        self.New_H=New_H
        self.New_W=New_W
        self.mean_bgr=mean_bgr 

    def transform(self,img,lbl):

        #img: PIL.Image, [h,w,c]
        #lbl: PIL.Image, [h,w]

        #for img
        img=self.resize(img)
        img=np.asarray(img)
        img=img[:,:,::-1]
        img=img.astype(np.float64)
        img-=self.mean_bgr
        img=img.transpose(2,0,1)
        img=torch.from_numpy(img).float()

        #for label
        lbl=self.resize(lbl)
        lbl=np.asarray(lbl).astype(np.int64)
        lbl.setflags(write=1)
        lbl[lbl>128]=255
        lbl[lbl<=128]=0
        lbl=torch.from_numpy(lbl).long()

        return img,lbl
        

        
        


