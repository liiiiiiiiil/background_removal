import os
import PIL.Image as Image
import numpy as np
from torchvision.transforms import Resize

root_dir='/mnt/disk1/lihao/person_br/datasets/icome_task2_data/clean_images/images'
img_names=os.listdir(os.path.join(root_dir))
count=len(img_names)
New_H,New_W=512,512
rs=Resize([New_H,New_W])


sum_array=np.zeros([New_H,New_W,3])
for img_name in img_names:
    img=Image.open(os.path.join(root_dir,'%s'%img_name))
    img=rs(img)
    img_array=np.asarray(img)[:,:,::-1]
    sum_array+=img_array

s1=np.sum(sum_array[:,:,0])/(New_H*New_W*count)
s2=np.sum(sum_array[:,:,1])/(New_H*New_W*count)
s3=np.sum(sum_array[:,:,2])/(New_H*New_W*count)

print(s1,s2,s3)


    

