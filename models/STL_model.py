import numpy as np
from .utils import get_blocks, create_optimizer, post_proc_losses
from .unet import UNet
from .unet_parts import *


class STL_model(UNet):
    def __init__(self, 
                 task_groups,
                 opt,
                 bilinear=True):
        super(STL_model, self).__init__(task_groups, opt, None, bilinear)
        self.per_batch_step = opt.per_batch_step
        self.one_optim_per_task = opt.one_optim_per_task
        self.learning_rate = opt.learning_rate
        self.active_task = opt.active_task
        
        # Optimizer
        self.optimizer = create_optimizer(opt.optimizer, self.parameters(), self.learning_rate)
            
    
    def optim_step(self, task=None):
        self.optimizer.step()
            
    def optim_zero_grad(self, task=None):
        self.optimizer.zero_grad()
            
    def train_step(self, 
                   data, 
                   gts, 
                   loss_func):
        # Forward pass
        logits = self.forward(data)
        task_losses = loss_func(logits, gts)
        loss = task_losses[self.active_task]
        task_losses = task_losses.detach()

        # Backward pass
        self.optim_zero_grad()
        loss.backward()
        self.optim_step()

        # Incr iter nb
        self.n_iter += 1
        preds = [elt.detach() for elt in logits]
            
        return task_losses, preds
    

    def test_step(self, data, gts, loss_func):
        # Forward pass
        logits = self.forward(data)
        task_losses = loss_func(logits, gts)
        loss = task_losses[self.active_task]
        
        return task_losses, logits

    
    def initialize(self, 
                   opt, 
                   device, 
                   model_dir, 
                   saver):
        super(STL_model, self).initialize(opt, 
                                          device, 
                                          model_dir)
        # Nothing if no load required
        if opt.recover:
            source_dir = os.path.join(opt.checkpoint_path, opt.reco_name) if opt.reco_name else model_dir
            ckpt_file = os.path.join(source_dir, opt.reco_type+'_weights.pth')
            ckpt = torch.load(ckpt_file, map_location=device)
            # Recovers optimizers if existing
            if 'optimizer_state_dict' in ckpt:
                if self.one_optim_per_task:
                    for k in range(len(ckpt['optimizer_state_dict'])):
                        self.optimizer[k].load_state_dict(ckpt['optimizer_state_dict'][str(k)])
                        print('Optimizer {} recovered from {}.'.format(k, ckpt_file))
                else:
                    self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                    print('Optimizer recovered from {}.'.format(ckpt_file))
            # Recovers saver
            saver.load()
