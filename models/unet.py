""" Full assembly of the parts to form the complete network """
import os
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
from .unet_parts import *
from sklearn.linear_model import LogisticRegression

        
class UNet(nn.Module):
    def __init__(self, 
                 task_groups,
                 opt,
                 partitioning=None,
                 bilinear=True):
        super(UNet, self).__init__()
        self.task_groups = task_groups
        self.n_tasks = len(task_groups)
        self.bilinear = bilinear
        self.n_iter = 0
        self.n_epoch = 0
        self.size = 18

        self.inc = DoubleConv(3, 
                              64, 
                              n_tasks=self.n_tasks)
        self.down1 = Down(64, 
                          128, 
                          n_tasks=self.n_tasks)
        self.down2 = Down(128, 
                          256, 
                          n_tasks=self.n_tasks)
        self.down3 = Down(256, 
                          512, 
                          n_tasks=self.n_tasks)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 
                          1024 // factor, 
                          n_tasks=self.n_tasks)
        self.up1 = Up(1024, 
                      512 // factor, 
                      n_tasks=self.n_tasks, 
                      bilinear=bilinear)
        self.up2 = Up(512, 
                      256 // factor, 
                      n_tasks=self.n_tasks, 
                      bilinear=bilinear)
        self.up3 = Up(256, 
                      128 // factor, 
                      n_tasks=self.n_tasks, 
                      bilinear=bilinear)
        self.up4 = Up(128, 
                      64, 
                      n_tasks=self.n_tasks, 
                      bilinear=bilinear)
        self.outcs = nn.ModuleList([OutConv(64, 1), OutConv(64, 1), OutConv(64, 1)])
        self.outfc = OutFC(1024 // factor, 1)
        self.clf = LogisticRegression(random_state=0, solver='lbfgs')

    def train_clf(self, vCDRs, classif_labs):
        self.clf.fit(vCDRs, classif_labs)
        
    def forward(self, x, task=None):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        if task == 0:
            return torch.sigmoid(self.outfc(x5))
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        if task:
            seg_logits = torch.sigmoid(self.outcs[task-1](x))
            return seg_logits
        else:
            return [torch.sigmoid(self.outfc(x5)), torch.sigmoid(self.outcs[0](x)), torch.sigmoid(self.outcs[1](x)), torch.sigmoid(self.outcs[2](x))]
    

    def initialize(self, 
                   opt, 
                   device, 
                   model_dir):

        # Nothing if no load required
        if opt.recover:
            source_dir = os.path.join(opt.checkpoint_path, opt.reco_name) if opt.reco_name else model_dir
            ckpt_file = os.path.join(source_dir, opt.reco_type+'_weights.pth')
            ckpt = torch.load(ckpt_file, map_location=device)
            
            # Gets what needed in the checkpoint
            pretrained_dict = {k:v for k,v in ckpt['model_state_dict'].items() if 'CONV' in k or 'BN' in k or 'FC' in k or 'outcs' in k or 'outfc' in k}
            
            # Loads the weights
            self.load_state_dict(pretrained_dict, strict=False)
            self.clf = ckpt['classifier']
            print('Weights and classifier recovered from {}.'.format(ckpt_file))
            
            # For recovering only
            if source_dir == model_dir:
                self.n_epoch = ckpt['epoch']
                self.n_iter = ckpt['n_iter'] + 1
                
        elif opt.pretrained:
            model_dict = {k: v for k, v in self.state_dict().items() if (not 'up' in k and not 'out' in k and not 'num_batches_tracked' in k)}
            url='https://download.pytorch.org/models/vgg13_bn-abd245e5.pth'
            pretrained_dict = model_zoo.load_url(url)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if (not 'classifier' in k and not 'running' in k)}
            new_pretrained_dict = {}
            pretrained_list = [(k, v) for k, v in pretrained_dict.items()] 
            model_list = [(k, v) for k, v in model_dict.items()] 
            for k in range(len(pretrained_dict.items())):
                kp, vp = pretrained_list[k]
                km, vm = model_list[k]
                if vp.shape == vm.shape:
                    new_pretrained_dict[km] = vp
                else:
                    new_pretrained_dict[km] = vm
            self.load_state_dict(new_pretrained_dict, strict=False)
        
        
    def checkpoint(self):
        # Prepares checkpoint
        ckpt = {'model_state_dict': self.state_dict(),
                'classifier': self.clf,
                'epoch': self.n_epoch,
                'n_iter': self.n_iter}
        return ckpt
