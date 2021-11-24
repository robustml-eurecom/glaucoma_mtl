import os
import cv2
import torch
import subprocess as sp
import numpy as np
from scipy.ndimage.measurements import label
from models import *

EPS = 1e-7

def check_gpu():
    """
    Selects an available GPU
    """
    available_gpu = -1
    ACCEPTABLE_USED_MEMORY = 500
    COMMAND = "nvidia-smi --query-gpu=memory.used --format=csv"
    _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
    memory_used_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
    memory_used_values = [int(x.split()[0]) for i, x in enumerate(memory_used_info)]
    for k in range(len(memory_used_values)):
        if memory_used_values[k]<ACCEPTABLE_USED_MEMORY:
            available_gpu = k
            break
    return available_gpu


def refine_seg(pred):
    '''
    Only retain the biggest connected component of a segmentation map.
    '''
    np_pred = pred.numpy()
        
    largest_ccs = []
    for i in range(np_pred.shape[0]):
        labeled, ncomponents = label(np_pred[i,:,:])
        bincounts = np.bincount(labeled.flat)[1:]
        if len(bincounts) == 0:
            largest_cc = labeled == 0
        else:
            largest_cc = labeled == np.argmax(bincounts)+1
        largest_cc = torch.tensor(largest_cc, dtype=torch.float32)
        largest_ccs.append(largest_cc)
    largest_ccs = torch.stack(largest_ccs)
    
    return largest_ccs
    
    
def select_model(opt, task_groups):
    """
    Select the model to use.
    """
    method = opt.method.upper()
    if method == 'MTL':
        model = MTL_model(task_groups, opt)
    if method == 'STL':
        model = STL_model(task_groups, opt)
    return model



class logger():
    """
    Simple logger, which display the wanted metrics in a proper format.
    """
    def __init__(self, metrics, to_disp=['auc', 'auc_unet', 'auc_vcdr', 'dsc_od', 'dsc_oc', 'vCDR_error', 'fov_error']):
        self.metrics=metrics
        self.to_disp=to_disp
        
    def log(self, n_epoch, n_iter, train_metrics, val_metrics, train_loss, val_loss):
        print('EVAL epoch {} iter {}: '.format(n_epoch, n_iter), ' '*50)
        for k in range(len(self.metrics)):
            if self.metrics[k] in self.to_disp:
                print('{} : {:.4f} / {:.4f}'.format(self.metrics[k], train_metrics[k]['reduced'], val_metrics[k]['reduced']))
        print('_'*50)
                  

def vertical_diameter(binary_segmentation):
    '''
    Get the vertical diameter from a binary segmentation.
    The vertical diameter is defined as the "fattest" area of the binary_segmentation parameter.
    '''
    # get the sum of the pixels in the vertical axis
    vertical_axis_diameter = np.sum(binary_segmentation, axis=1)

    # pick the maximum value
    diameter = np.max(vertical_axis_diameter, axis=1)

    # return it
    return diameter


                
def vertical_cup_to_disc_ratio(od, oc):
    '''
    Compute the vertical cup-to-disc ratio from a given labelling map.
    '''
    # compute the cup diameter
    cup_diameter = vertical_diameter(oc)
    # compute the disc diameter
    disc_diameter = vertical_diameter(od)

    return cup_diameter / (disc_diameter + EPS)


def get_vCDRs(preds, gts):
    pred_od = preds[1][:,0,:,:].cpu().numpy()
    pred_oc = preds[2][:,0,:,:].cpu().numpy()
    gt_od = gts[1][:,0,:,:].cpu().numpy()
    gt_oc = gts[2][:,0,:,:].cpu().numpy()
    pred_vCDR = vertical_cup_to_disc_ratio(pred_od, pred_oc)
    gt_vCDR = vertical_cup_to_disc_ratio(gt_od, gt_oc)
    return pred_vCDR, gt_vCDR

    