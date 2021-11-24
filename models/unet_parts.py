""" Parts of the U-Net model """
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class ConvLayer(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size=3, 
                 padding=1, 
                 norm_layer=True, 
                 relu=True,
                 n_tasks=None):
        
        super(ConvLayer, self).__init__()
        modules = [('CONV', nn.Conv2d(in_channels, 
                                      out_channels, 
                                      kernel_size=kernel_size, 
                                      padding=padding))]
        if norm_layer:
            modules.append(('BN', nn.BatchNorm2d(num_features=out_channels, 
                                                 track_running_stats=False)))
        if relu:
            modules.append(('relu', nn.ReLU(inplace=True)))
            
        self.conv_block = nn.Sequential(OrderedDict(modules))

    def forward(self, x):
        return self.conv_block(x)
    
    def get_weight(self):
        return self.conv_block[0].weight
        
    def get_routing_block(self):
        return self.conv_block[-2]
    
    def get_routing_masks(self):
        mapping = self.conv_block[-2].unit_mapping.detach().cpu().numpy()
        tested = self.conv_block[-2].tested_tasks.detach().cpu().numpy()
        return mapping, tested 
        

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 mid_channels=None,
                 n_tasks=None, 
                ):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
            
        self.double_conv = nn.Sequential(
            ConvLayer(in_channels, 
                      mid_channels, 
                      n_tasks=n_tasks),
            ConvLayer(mid_channels, 
                      out_channels, 
                      n_tasks=n_tasks),
        )
            
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 n_tasks=None, 
                ):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, 
                       out_channels, 
                       n_tasks=n_tasks, 
                      )
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, 
                 in_channels, 
                 out_channels,
                 bilinear=True,
                 n_tasks=None, 
                 ):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, 
                                   out_channels, 
                                   mid_channels=in_channels // 2,
                                   n_tasks=n_tasks, 
                                  )
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, 
                                   out_channels, 
                                   n_tasks=n_tasks, 
                                  )


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)



class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
class OutFC(nn.Module):
    def __init__(self, in_channels, out_channels, p=0.8):
        super(OutFC, self).__init__()
        self.pool = nn.AdaptiveMaxPool2d(1)
        self.dropout = nn.Dropout(p=p)
        self.fc = nn.Linear(in_channels, out_channels, bias=True)
    
    def forward(self, x):
        flat_pool = self.pool(x).squeeze(2).squeeze(2)
        return self.fc(self.dropout(flat_pool)).squeeze(1)
    