import torch
import torch.nn as nn

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1,padding=0):
        super(BasicConv2d,self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.bn   = nn.BatchNorm2d(out_planes, eps=0.001, momentum=0.1, affine=True)
        self.elu = nn.ELU(alpha=1)
    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.elu(x)
        return x

class InceptionBlock(nn.Module):
    def __init__(self,in_channels):
        super(InceptionBlock, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels,32,kernel_size=1)

        self.branch5x5_1 = BasicConv2d(in_channels,16,kernel_size=1)
        self.branch5x5_2 = BasicConv2d(16,32,kernel_size=5,padding=2)

        self.branch3x3_1 = BasicConv2d(in_channels,16,kernel_size=1)
        self.branch3x3_2 = BasicConv2d(16,32,kernel_size=3,padding=1)
        self.branch3x3_3 = BasicConv2d(32,32,kernel_size=3,padding=1)

        self.branch_pool_1 = nn.AvgPool2d(kernel_size=3,stride=1,padding=1)
        self.branch_pool_2 = BasicConv2d(in_channels,32,kernel_size=1)

    def forward(self,x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)
 
        branchpool = self.branch_pool_1(x)
        branchpool = self.branch_pool_2(branchpool)

        outputs = [branch1x1,branch5x5,branch3x3,branchpool]
        return torch.cat(outputs,1)


class SFHNet(nn.Module):
    def __init__(self,batch_size):
        self.batch_size = batch_size
        super(SFHNet,self).__init__()

        self.features = nn.Sequential(
                        nn.Conv2d(in_channels=1,out_channels=16,kernel_size=(3,3),stride=1,padding=1),
                        nn.ELU(alpha=1),
                        nn.AvgPool2d(2),
                        nn.Conv2d(in_channels=16,out_channels=32,kernel_size=(3,3),stride=1,padding=1),
                        nn.ELU(alpha=1),
                        nn.AvgPool2d(2),
                        InceptionBlock(in_channels=32),
                        nn.Conv2d(in_channels=128,out_channels=256,kernel_size=(3,3),stride=1,padding=1),
                        nn.ELU(alpha=1),
                        nn.AvgPool2d(2),
                        )
   
        self.recover = nn.Sequential(
           nn.Conv2d(in_channels=256,out_channels=128,kernel_size=(3,3),stride=1,padding=1),
           nn.ELU(alpha=1),
           nn.Upsample(scale_factor=2,mode='bilinear'),
           nn.Conv2d(in_channels=128,out_channels=64,kernel_size=(3,3),stride=1,padding=1),
           nn.ELU(alpha=1),
           nn.Upsample(scale_factor=2,mode='bilinear'),
           nn.Conv2d(in_channels=64,out_channels=32,kernel_size=(3,3),stride=1,padding=1),
           nn.ELU(alpha=1),
           nn.Upsample(scale_factor=2,mode='bilinear'),
           nn.Conv2d(in_channels=32,out_channels=1 ,kernel_size=(3,3),stride=1,padding=1),
        )
 
        self.resizer = nn.Sequential(
           nn.Linear(in_features=256*10*10, out_features=128*10*10),	
           nn.ELU(alpha=1),
           nn.Linear(in_features=128*10*10, out_features=64*10*10),
           nn.ELU(alpha=1),
           nn.Linear(in_features=64*10*10, out_features=32*10*10),
           nn.ELU(alpha=1),
           nn.Linear(in_features=32*10*10, out_features=1560),
           nn.ELU(alpha=1)
        )

        self.classifier = nn.Sequential(
           nn.Upsample(scale_factor=2,mode='bilinear'),
           nn.Conv2d(in_channels=20, out_channels=5, kernel_size=(3,3),stride=1,padding=1),
           nn.ELU(alpha=1),
           nn.Upsample(scale_factor=2,mode='bilinear'),
           nn.Conv2d(in_channels=5, out_channels=1, kernel_size=(3,3),stride=1,padding=1),
           nn.ELU(alpha=1)
        )    
    
        
    def forward(self,x):
        x = self.features(x)        
        aux = x

        x = x.view(self.batch_size,-1)
        x = self.resizer(x)
        x = x.view(self.batch_size,20,6,13)
        x = self.classifier(x)
        x = x.view(self.batch_size,1,1248)
        
        aux = aux.view(self.batch_size,256,10,10)
        aux = self.recover(aux)
       
        return x,aux



        

