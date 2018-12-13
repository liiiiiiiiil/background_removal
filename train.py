import torch
from utils import cross_entropy2d


def train_epoch(model,optimizer,criterion,train_loader,infos,args):
    iteration=infos['iteration'] 
    epoch=infos['epoch']
    

    for i,batch in enumerate(train_loader):
        iteration+=1

        optimizer.zero_grad()

        inputs=batch[0]
        labels=batch[1]
        if args.use_cuda:
            inputs=inputs.cuda()
            labels=labels.cuda()

        outputs=model(inputs)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        
        if iteration%args.checkpoint_every==0 and args.checkpoint_every>0:
            print('Epoch:{},iteration:{},train_loss:{}'.format(epoch,iteration,loss))
            infos['train_loss'].append(loss)

        infos['iteration']=iteration
        

             
            
            
            
            
            

        
        
        
        
        

        

        
        

        
        

