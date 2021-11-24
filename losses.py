import torch
import torch.nn as nn
import torch.nn.functional as F



class RefugeLoss(nn.Module):
    def __init__(self, task_groups, opt):
        super(RefugeLoss, self).__init__()
        self.task_losses = [focal_loss,
                            mean_BCE_loss,
                            mean_BCE_loss,
                            fov_loss]
        self.task_groups = task_groups
                
    def forward(self, preds, gts, task=None):
        if task==None:
            task_losses = [self.task_losses[k](preds[k], gts[k]) for k in range(len(self.task_losses))]
            return torch.stack(task_losses)
        
        else:
            task_loss = self.task_losses[task](preds, gts[task])
            return task_loss
        

        
def mean_BCE_loss(pred, gt):
    loss = F.binary_cross_entropy(pred, gt, reduction='mean')
    return loss

def fov_loss(pred, gt):
    fov, pdf = gt
    loss = F.l1_loss(pred[:,0,:,:], pdf, reduction='mean')
    return loss


def focal_loss(pred, gt, alpha=1, gamma=2):
    BCE_loss = F.binary_cross_entropy(pred, gt, reduction='none')
    pt = torch.exp(-BCE_loss)
    F_loss = alpha * (1-pt)**gamma * BCE_loss

    return torch.mean(F_loss)
