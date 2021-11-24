import os
import json
import scipy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from scipy.stats import multivariate_normal

class RefugeDataset(Dataset):
    '''
    Loads all data samples once and for all into memory. Can speed up
    the training depending on the computer setup.
    '''
    def __init__(self, root_dir, split='train', output_size=(256,256)):
        # Define attributes
        self.output_size = output_size
        self.root_dir = root_dir
        self.split = split
        
        # Load data index
        with open(os.path.join(self.root_dir, self.split, 'index.json')) as f:
            self.index = json.load(f)
            
        # Sample lists
        self.images = []
        self.labs = []
        self.ods = []
        self.ocs = []
        self.fovs = []
        self.pdfs = []
        
        # Loading
        for k in range(len(self.index)):
            print('Loading {} sample {}/{}...'.format(split, k, len(self.index)), end='\r')
            base_height = self.index[str(k)]['Size_Y']
            base_width = self.index[str(k)]['Size_X']
            
            # Image
            img_name = os.path.join(self.root_dir, self.split, 'images', self.index[str(k)]['ImgName'])
            img = Image.open(img_name).convert('RGB')
            w,h = img.size
            img = transforms.functional.resize(img, self.output_size, interpolation=Image.BILINEAR)
            img = transforms.functional.to_tensor(img)
            self.images.append(img)
 

            # Label
            lab = torch.tensor(self.index[str(k)]['Label'], dtype=torch.float32)
            self.labs.append(lab)

            # Seg
            seg_name = os.path.join(self.root_dir, self.split, 'seg', self.index[str(k)]['ImgName'].split('.')[0]+'.bmp')
            seg = np.array(Image.open(seg_name)).copy()
            seg = 255. - seg
            od = Image.fromarray((seg>=127.).astype(np.float32))
            oc = Image.fromarray((seg>=250.).astype(np.float32))
            od = transforms.functional.resize(od, self.output_size, interpolation=Image.NEAREST)
            oc = transforms.functional.resize(oc, self.output_size, interpolation=Image.NEAREST)
            od = transforms.functional.to_tensor(od)
            oc = transforms.functional.to_tensor(oc)
            self.ods.append(od)
            self.ocs.append(oc)


            
            # Fovea
            f_x = self.index[str(k)]['Fovea_X']/base_width*self.output_size[1]
            f_y = self.index[str(k)]['Fovea_Y']/base_height*self.output_size[0]
            x, y = np.mgrid[0:self.output_size[1]:1, 0:self.output_size[0]:1]
            pos = np.dstack((x, y))
            cov = 50
            rv = multivariate_normal([f_y, f_x], [[cov,0],[0,cov]])
            pdf = rv.pdf(pos)
            pdf = pdf/np.max(pdf)
            pdf = transforms.functional.to_tensor(Image.fromarray(pdf))[0,:,:]
            (f_y, f_x) = scipy.ndimage.measurements.center_of_mass(pdf.numpy())
            fov = torch.FloatTensor([f_x, f_y])
            self.fovs.append(fov)
            self.pdfs.append(pdf)
            
        print('Succesfully loaded {} dataset.'.format(split) + ' '*50)
            
            
    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        # Image
        img = self.images[idx]
    
        # Label
        lab = self.labs[idx]

        # Segmentation masks
        od = self.ods[idx]
        oc = self.ocs[idx]

        # Fovea localization
        fov = self.fovs[idx]
        pdf = self.pdfs[idx]

        return img, [lab, od, oc, (fov, pdf)]
    
    
    
    
    
    
    
class RefugeDataset2(Dataset):
    '''
    Usual on-line loading dataset. More memory efficient.
    '''
    def __init__(self, root_dir, split='train', output_size=(256,256)):
        # Define attributes
        self.output_size = output_size
        self.root_dir = root_dir
        self.split = split
        
        # Load data index
        with open(os.path.join(self.root_dir, self.split, 'index.json')) as f:
            self.index = json.load(f)
            
            
    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        base_height = self.index[str(idx)]['Size_Y']
        base_width = self.index[str(idx)]['Size_X']
        
        # Image
        img_name = os.path.join(self.root_dir, self.split, 'images', self.index[str(idx)]['ImgName'])
        img = Image.open(img_name).convert('RGB')
        w,h = img.size
        img = transforms.functional.resize(img, self.output_size, interpolation=Image.BILINEAR)
        img = transforms.functional.to_tensor(img)
    
        # Label
        lab = torch.tensor(self.index[str(idx)]['Label'], dtype=torch.float32)

        # Segmentation masks
        seg_name = os.path.join(self.root_dir, self.split, 'seg', self.index[str(idx)]['ImgName'].split('.')[0]+'.bmp')
        seg = np.array(Image.open(seg_name)).copy()
        seg = 255. - seg
        od = Image.fromarray((seg>=127.).astype(np.float32))
        oc = Image.fromarray((seg>=250.).astype(np.float32))
        od = transforms.functional.resize(od, self.output_size, interpolation=Image.NEAREST)
        oc = transforms.functional.resize(oc, self.output_size, interpolation=Image.NEAREST)
        od = transforms.functional.to_tensor(od)
        oc = transforms.functional.to_tensor(oc)
        
        # Fovea localization
        f_x = self.index[str(idx)]['Fovea_X']/base_width*self.output_size[1]
        f_y = self.index[str(idx)]['Fovea_Y']/base_height*self.output_size[0]
        x, y = np.mgrid[0:self.output_size[1]:1, 0:self.output_size[0]:1]
        pos = np.dstack((x, y))
        cov = 50
        rv = multivariate_normal([f_y, f_x], [[cov,0],[0,cov]])
        pdf = rv.pdf(pos)
        pdf = pdf/np.max(pdf)
        pdf = transforms.functional.to_tensor(Image.fromarray(pdf))[0,:,:]
        (f_y, f_x) = scipy.ndimage.measurements.center_of_mass(pdf.numpy())
        fov = torch.FloatTensor([f_x, f_y])

        return img, [lab, od, oc, (fov, pdf)]