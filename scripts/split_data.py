import os

data_dir='./'
num_for_train=4500

imgs_dir_path=os.path.join(data_dir,'clean_images/images')
labels_dir_path=os.path.join(data_dir,'clean_images/profiles')

img_names=os.listdir(imgs_dir_path)
print('There are {} image'.format(len(img_names)))
label_names=os.listdir(labels_dir_path)
print('There are {} labels'.format(len(label_names)))
assert len(img_names)==len(label_names),'Num of img != Num of labels'

train_names=img_names[:num_for_train]
val_names=img_names[num_for_train:]

print('There are {} images for train and {} images for val'.format(len(train_names),len(val_names)))

f=open(data_dir+'train.txt','a')
for img in train_names:
    f.write(img+'\n')
f.close

f=open(data_dir+'val.txt','a')
for img in val_names:
    f.write(img+'\n')
f.close

