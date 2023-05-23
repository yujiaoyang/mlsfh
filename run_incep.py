import numpy as np
import os
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau,StepLR
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from datetime import datetime
from model_inception import SFHNet
from torch.utils.tensorboard import SummaryWriter
import configparser

config = configparser.ConfigParser()
config.read('config.ini')

dirs = config['DEFAULT']['PATH_PACk']+config['OBJECT']['NAME'] + '_' + config['OBJECT']['FIELD'] + '/'
CONFIG_EPOCHS     = int(config['TRAIN']['EPOCHS'])
CONFIG_BATCH_SIZE = int(config['TRAIN']['BATCH_SIZE'])
CONFIG_CUDA_NUM   = int(config['TRAIN']['CUDA_NUM'])
CONFIG_LR         = float(config['TRAIN']['LR'])

class SFHData(Dataset):
    def __init__(self,filename):
        super(SFHData, self).__init__()

        data = np.load(filename)
        imgs = torch.from_numpy(data['imgs'])
        targets = torch.from_numpy(data['targets']-1)

        self.imgs    = np.transpose(imgs,(0,3,1,2))
        self.targets = targets

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self,idx):
        img    = self.imgs[idx]
        target = torch.reshape(self.targets[idx],(1,1248))
        return img, target

train_data = SFHData(filename='train.npz')
valid_data = SFHData(filename='valid.npz')
train_loader = DataLoader(dataset = train_data,batch_size=CONFIG_BATCH_SIZE,shuffle=True ,num_workers=40)
valid_loader = DataLoader(dataset = valid_data,batch_size=CONFIG_BATCH_SIZE,shuffle=False,num_workers=40)
print('num_of_trainData:',len(train_data))
print('num_of_validData:',len(valid_data))


###################################################################
#load model
if not os.path.exists(dirs + 'run'):
    os.makedirs(dirs + 'run')

model = SFHNet(batch_size = CONFIG_BATCH_SIZE)
model.cuda(CONFIG_CUDA_NUM)
model.train()
loss_fn  =  nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=CONFIG_LR)
scheduler = ReduceLROnPlateau(optimizer,'min',factor=0.95,patience=20,min_lr=0)
#scheduler = StepLR(optimizer, step_size=10,gamma=0.9)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('run/trainer_{}'.format(timestamp))
epoch_number = 0

savefreq = 100
for epoch in range(CONFIG_EPOCHS):
    print ('EPOCH: {}'.format(epoch_number+1))
    model.train(True)
    
    running_loss  = 0
    running_loss1 = 0
    running_loss2 = 0
    for batch_idx, traindata in enumerate(train_loader):
        imgs, targets = traindata
        optimizer.zero_grad()
        outputs = model(imgs.cuda(CONFIG_CUDA_NUM))

        loss1 = loss_fn(outputs[0],targets.cuda(CONFIG_CUDA_NUM))
        loss2 = loss_fn(outputs[1],imgs.cuda(CONFIG_CUDA_NUM))

        loss = 100*loss1+loss2
        loss.backward()
        optimizer.step()
        running_loss  += loss.item()
        running_loss1 += loss1.item()
        running_loss2 += loss2.item()
    
    avg_loss  = running_loss/(batch_idx+1)
    avg_loss1 = running_loss1/(batch_idx+1)
    avg_loss2 = running_loss2/(batch_idx+1)

    model.train(False)

    running_vloss  = 0
    running_vloss1 = 0
    running_vloss2 = 0
    for batch_idx, validdata in enumerate(valid_loader):
        vimgs, vtargets = validdata
        voutputs = model(vimgs.cuda(CONFIG_CUDA_NUM))

        vloss1 = loss_fn(voutputs[0],vtargets.cuda(CONFIG_CUDA_NUM))
        vloss2 = loss_fn(voutputs[1],vimgs.cuda(CONFIG_CUDA_NUM))

        vloss = 100*vloss1+vloss2
        running_vloss  += vloss.item()
        running_vloss1 += vloss1.item()
        running_vloss2 += vloss2.item()
    
    avg_vloss  = running_vloss  / (batch_idx+1)
    avg_vloss1 = running_vloss1 / (batch_idx+1)
    avg_vloss2 = running_vloss2 / (batch_idx+1)

    curr_lr = optimizer.param_groups[0]['lr']

    writer.add_scalars('Training vs. Validation Loss',
                      {'Training_all' : avg_loss,  'Validation_all': avg_vloss,
                       'Training_par' : avg_loss1, 'Validation_par': avg_vloss1,
                       'Training_cmd' : avg_loss2, 'Validation_cmd': avg_vloss2,
                       'lr'           : curr_lr},epoch_number+1)
    writer.flush()
    
    if epoch % savefreq == savefreq-1:
        save_info = {'epoch' : epoch,
    		     'model_state_dict':model.state_dict(),
                     'optimizer_state_dict':optimizer.state_dict(),
                     'train_loss':avg_loss,
    		     'valid_loss':avg_vloss,}
        model_path ='model_{}'.format(epoch_number+1)
        torch.save(save_info,model_path)

    epoch_number += 1
    print('EPOCH: {} train avg_loss_all : {} valid avg_loss_all : {}'.format(epoch_number,avg_loss ,avg_vloss ))
    print('EPOCH: {} train avg_loss_par : {} valid avg_loss_par : {}'.format(epoch_number,avg_loss1,avg_vloss1))
    print('EPOCH: {} train avg_loss_cmd : {} valid avg_loss_cmd : {}'.format(epoch_number,avg_loss2,avg_vloss2))
    print('LR   : {}'.format(curr_lr))

    scheduler.step(avg_vloss)


