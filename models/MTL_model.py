import numpy as np
from .utils import get_blocks, create_optimizer, post_proc_losses
from .unet import UNet
from .unet_parts import *


class MTL_model(UNet):
    def __init__(self, 
                 task_groups,
                 opt,
                 bilinear=True):
        super(MTL_model, self).__init__(task_groups, opt, None, bilinear)
        self.per_batch_step = opt.per_batch_step
        self.one_optim_per_task = opt.one_optim_per_task
        self.learning_rate = opt.learning_rate
        
        # Optimizer
        if self.one_optim_per_task and not self.per_batch_step:
            self.optimizer = []
            for k in range(len(task_groups)):
                self.optimizer.append(create_optimizer(opt.optimizer, self.parameters(), self.learning_rate))
        else:
            self.optimizer = create_optimizer(opt.optimizer, self.parameters(), self.learning_rate)
            
    
    def optim_step(self, task=None):
        if self.one_optim_per_task and not self.per_batch_step:
            self.optimizer[task].step()
        else:
            self.optimizer.step()
            
    def optim_zero_grad(self, task=None):
        if self.one_optim_per_task and not self.per_batch_step:
            self.optimizer[task].zero_grad()
        else:
            self.optimizer.zero_grad()
            
    def train_step(self, 
                   data, 
                   gts, 
                   loss_func):
        # Tasks optimized altogether
        if self.per_batch_step:
            # Forward pass
            logits = self.forward(data)
            task_losses = loss_func(logits, gts)
            loss = task_losses.sum()
            task_losses = task_losses.detach()

            # Backward pass
            self.optim_zero_grad()
            loss.backward()
            self.optim_step()
            
            # Incr iter nb
            self.n_iter += 1
            preds = [elt.detach() for elt in logits]
            
        
        # Tasks optimized separately
        else:
            task_losses = [None]*self.n_tasks
            preds = [None]*self.n_tasks
            
            # Inference
            for task in range(len(self.task_groups)):
                # Forward pass
                logits = self.forward(data, task=task)
                task_loss = loss_func(logits, gts, task=task)

                # Backward pass
                self.optim_zero_grad(task)
                task_loss.backward()
                self.optim_step(task)
                task_losses[task] = task_loss.detach()
                preds[task] = logits.detach()

            # Incr iter nb
            self.n_iter += 1
            task_losses = torch.stack(task_losses)
            
        return task_losses, preds
    

    def test_step(self, data, gts, loss_func):
        # Forward pass
        logits = self.forward(data)
        task_losses = loss_func(logits, gts)
        loss = task_losses.sum()
        
        return task_losses, logits

    
    def initialize(self, 
                   opt, 
                   device, 
                   model_dir, 
                   saver):
        super(MTL_model, self).initialize(opt, 
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
