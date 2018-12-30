import torch 
import numpy as np
from misc.utils import iou,pixel_acc,save2image

def val_epoch(model,criterion,val_loader,infos,args):

    model.eval()    
    epoch=infos['epoch']
    losses=[]
    total_ious=[]
    pixel_accs=[]
    for i,batch in enumerate(val_loader):
        if i==args.max_val_iterations:
            break
        inputs=batch[0]
        labels=batch[1]
        
        if args.use_cuda:
            inputs=inputs.cuda()
            labels=labels.cuda()
        with torch.no_grad(): 
            outputs=model(inputs)
            loss=criterion(outputs,labels)
        losses.append(loss.item())
        outputs=outputs.data.cpu().numpy()
        N,_,h,w= outputs.shape 
        pred=outputs.transpose(0,2,3,1).reshape(-1,args.n_class).argmax(axis=1).reshape(N,h,w)
        targets=labels.data.cpu().numpy().reshape(N,h,w)

        for p,t in zip(pred,targets):
            total_ious.append(iou(p,t,args.n_class))
            pixel_accs.append(pixel_acc(p,t))

    val_loss=np.mean(losses)
    total_ious=np.array(total_ious).T
    ious=np.nanmean(total_ious,axis=1)
    pixel_accs=np.array(pixel_accs).mean()

    print("Epoch:{},val_loss:{},pix_acc:{},meanIoU:{},IoUs:{}".format(
        epoch,val_loss,pixel_accs,np.nanmean(ious),ious
        ))

    infos['val_loss'].append((epoch,val_loss))
    infos['mean_Pixel'].append((epoch,pixel_accs))
    infos['meanIU'].append((epoch,ious))
    model.train()

def eval_own_images(model,data_loader,args):

    model.eval()
    save_path=args.image_save_path
    for batch in data_loader:
        img_names=batch[0]
        imgs=batch[1]
        if args.use_cuda:
            imgs=imgs.cuda()

        with torch.no_grad():
            outputs=model(imgs)
        
        outputs=outputs.data.cpu().numpy()
        save2image(outputs,img_names,save_path)

    print('All {} images are save in:{}'.format(len(data_loader),args.image_save_path))

        

    


    
        

        


        
        
        
        
        
    
