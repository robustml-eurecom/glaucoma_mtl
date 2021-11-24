import scipy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score, roc_curve

from utils import refine_seg

EPS = 1e-7

class Metric():
    """
    Basic metric accumulator of chosen size.
    """
    def __init__(self, labels=False, ignore=-1., reduce_type=['mean']):
        self.acc = []
        self.reduce_type = reduce_type
        if labels:
            self.labels = []
        else:
            self.labels = None
        self.ignore = -1.
        
    def incr(self, value, labels=None):
        if not isinstance(value, list):
            value = [value]
        for k in range(len(value)):
            if value[k] != self.ignore:
                if labels:
                    if labels[k] != self.ignore:
                        self.acc.append(float(value[k]))
                        self.labels.append(float(labels[k]))
                else:
                    self.acc.append(float(value[k]))
                
    def result(self):
        res = {}
        if 'none' in self.reduce_type:
            res['values'] = self.acc
            if self.labels:
                res['labels'] = self.labels
                
        if self.labels:
            if 'auc' in self.reduce_type:
                res['reduced'] = classif_eval(self.acc, self.labels)
            
        else:
            if 'mean' in self.reduce_type:
                res['reduced'] = np.mean(self.acc)
            elif 'sum' in self.reduce_type:
                res['reduced'] = np.sum(self.acc)
        return res
    
    def reset(self):
        self.acc = []
        if self.labels:
            self.labels = []


class RefugeMetrics():
    """
    Refuge metric accumulator.
    """
    def __init__(self, task_groups, opt):
        self.task_groups = task_groups
        self.n_tasks = len(task_groups)
        self.metrics = []
        self.metrics_names = []
        self.classif_preds = []
        self.classif_gts = []
        
        # Adds all NYU metrics
        self.add_metric('auc_unet', True, reduce_type=['auc'])
        self.add_metric('auc_vcdr', True, reduce_type=['auc'])
        self.add_metric('auc', True, reduce_type=['auc'])
        self.add_metric('dsc_od', reduce_type=['mean'])
        self.add_metric('dsc_oc', reduce_type=['mean'])
        self.add_metric('vCDR_error', reduce_type=['mean'])
        self.add_metric('fov_error', reduce_type=['mean'])
        
        # Set metrics to 0
        self.reset()
    
    def link_metric(self, source_metric, target_metrics, link_type='mean'):
        self.metrics[self.metrics_names.index(source_metric)].link([self.metrics[self.metrics_names.index(elt)] for elt in target_metrics], link_type)
        
    def add_metric(self, name, labels=False, reduce_type='mean'):
        self.metrics.append(Metric(labels=labels, reduce_type=reduce_type))
        self.metrics_names.append(name)
        
    def incr_metric(self, name, value, labels=None):
        self.metrics[self.metrics_names.index(name)].incr(value, labels=labels)

    def reset(self):
        self.n_samples = 0
        for met in self.metrics:
            met.reset()
                
    # def incr(self, preds, gts):
    def incr(self, preds, gts, pred_vCDR, gt_vCDR):
        device = preds[0].device
        pred_od = refine_seg((preds[1]>=0.5).type(torch.int8).cpu()).to(device)
        pred_oc = refine_seg((preds[2]>=0.5).type(torch.int8).cpu()).to(device)
        gt_od = gts[1].type(torch.int8)
        gt_oc = gts[2].type(torch.int8)
        dsc_od = compute_dice_coef(pred_od, gt_od)
        dsc_oc = compute_dice_coef(pred_oc, gt_oc)
        vCDR_error = compute_vCDR_error(pred_vCDR, gt_vCDR)

        self.incr_metric('auc_unet', preds[0].cpu().numpy().tolist(), gts[0].cpu().numpy().tolist())
        self.incr_metric('auc_vcdr', preds[-1].cpu().numpy().tolist(), gts[0].cpu().numpy().tolist())
        self.incr_metric('auc', ((preds[0]+preds[-1])/2).cpu().numpy().tolist(), gts[0].cpu().numpy().tolist())
        self.incr_metric('dsc_od', dsc_od.item())
        self.incr_metric('dsc_oc', dsc_oc.item())
        self.incr_metric('vCDR_error', vCDR_error)
        self.incr_metric('fov_error', fov_error(preds[3].cpu().detach().numpy(), gts[3][0].cpu().numpy()))
                
        self.n_samples += 1
                
        
    def result(self):
        res = []
        for met in self.metrics:
            res.append(met.result())
        return res
    


def compute_dice_coef(input, target):
    '''
    Compute dice score metric.
    '''
    batch_size = input.shape[0]
    return sum([dice_coef_sample(input[k,:,:], target[k,:,:]) for k in range(batch_size)])/batch_size


def dice_coef_sample(input, target):
    iflat = input.contiguous().view(-1)
    tflat = target.type(torch.float32).contiguous().view(-1)
    intersection = (iflat * tflat).sum()
    res = (2. * intersection) / (iflat.sum() + tflat.sum())
    return res


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
    cup_diameter = vertical_diameter(oc)
    disc_diameter = vertical_diameter(od)
    return cup_diameter / (disc_diameter + EPS)


def compute_vCDR_error(pred_vCDR, gt_vCDR):
    '''
    Compute vCDR prediction error, along with predicted vCDR and ground truth vCDR.
    '''
    vCDR_err = np.mean(np.abs(gt_vCDR - pred_vCDR))
    return vCDR_err


def classif_eval(classif_preds, classif_gts):
    '''
    Compute AUC classification score.
    '''
    auc = roc_auc_score(classif_gts, classif_preds)
    return auc


def fov_error(seg_fov, fov_coord):
    mass_centers = [scipy.ndimage.measurements.center_of_mass(seg_fov[k,0,:,:]) for k in range(seg_fov.shape[0])]
    mass_centers = np.array([[elt[1], elt[0]] for elt in mass_centers])
    err = np.sqrt(np.sum((fov_coord-mass_centers)**2, axis=1)).mean()
    return err
    
