import os
import json
import numpy as np
import torch

def merge_logs(old_logs, new_logs):
    """
    Merges two log dicts.
    """
    for new_branch_name in new_logs.keys():
        if not new_branch_name in old_logs.keys() or not  isinstance(old_logs[new_branch_name], dict):
            old_logs[new_branch_name] = new_logs[new_branch_name]
        elif isinstance(old_logs[new_branch_name], dict):
            merge_logs(old_logs[new_branch_name], new_logs[new_branch_name])



class Saver():
    """
    Saving object for both metrics, weights and config of a model.
    """
    def __init__(self, path, args=None, iter=0):
        self.save_dir = path
        self.logs_dict = {}
        self.logs_file = os.path.join(self.save_dir,'logs.json')
        self.STL = False
        self.best_error = np.inf
        if args:
            self.config_dict = vars(args)
            if args.method.upper() == 'STL':
                self.STL = True
                self.active_task = args.active_task
                if self.active_task!=3:
                    self.best_error = 0.
        self.config_file = os.path.join(self.save_dir,'config.json')
        self.best_error_weights_file = os.path.join(self.save_dir,'best_error_weights.pth')
        self.metrics = []
        
    def add_metrics(self, metrics):
        """
        Adds metrics to the saver.
        """
        self.metrics += metrics
        
    def log(self, model, task_groups, epoch, n_iter, train_metrics, val_metrics, train_losses, val_losses, optimizer=None):
        """
        Adds new values to the saver, and eventually saves them.
        """
        # New iter dict
        iter_dict = {}
        
        # Metrics
        for k in range(len(self.metrics)):
            iter_dict[self.metrics[k]] = {'train': train_metrics[k], 
                                          'val': val_metrics[k]}
            
        # Losses
        for k in range(len(task_groups)):
            group_type = task_groups[k]['type']
            iter_dict['{}_loss'.format(group_type)] = {'train': float(train_losses[k]),
                                                       'val': float(val_losses[k])}
        iter_dict['loss'] = {'train': float(train_losses.sum()), 
                             'val': float(val_losses.sum())}
        

        # Update logs_dict
        if not str(n_iter) in self.logs_dict.keys():
            self.logs_dict[str(n_iter)] = iter_dict
        else:
            merge_logs(self.logs_dict[str(n_iter)], iter_dict)
        
        # Write JSON
        with open(self.logs_file, 'w') as f:
            json.dump(self.logs_dict, f)  
        
        # Checkpoint
        self.checkpoint(model, epoch, n_iter, iter_dict, optimizer)
        
            
        
    def checkpoint(self, model, epoch, n_iter, iter_dict, optimizer=None):
        """
        Applies different types of checkpoints.
        """
        # Prepares checkpoint
        ckpt = model.checkpoint()
        if optimizer:
            if isinstance(optimizer, list):
                ckpt['optimizer_state_dict'] = {}
                for k in range(len(optimizer)):
                    ckpt['optimizer_state_dict'][str(k)] = optimizer[k].state_dict()
            else:
                ckpt['optimizer_state_dict'] = optimizer.state_dict()

        # Saves best loss, if reached
        if self.STL:
            if self.active_task==0:
                ref_met = 'auc_unet'
                if iter_dict[ref_met]['val']['reduced'] > self.best_error:
                    print('Best error.')
                    torch.save(ckpt, self.best_error_weights_file)
                    self.best_error = iter_dict[ref_met]['val']['reduced']
                    self.config_dict['best_error'] = self.best_error
            elif self.active_task==1:
                ref_met = 'dsc_od'
                if iter_dict[ref_met]['val']['reduced'] > self.best_error:
                    print('Best error.')
                    torch.save(ckpt, self.best_error_weights_file)
                    self.best_error = iter_dict[ref_met]['val']['reduced']
                    self.config_dict['best_error'] = self.best_error
            elif self.active_task==2:
                ref_met = 'dsc_oc'
                if iter_dict[ref_met]['val']['reduced'] > self.best_error:
                    print('Best error.')
                    torch.save(ckpt, self.best_error_weights_file)
                    self.best_error = iter_dict[ref_met]['val']['reduced']
                    self.config_dict['best_error'] = self.best_error
            elif self.active_task==3:
                ref_met = 'fov_error'
                if iter_dict[ref_met]['val']['reduced'] < self.best_error:
                    print('Best error.')
                    torch.save(ckpt, self.best_error_weights_file)
                    self.best_error = iter_dict[ref_met]['val']['reduced']
                    self.config_dict['best_error'] = self.best_error
                    
        else:
            if iter_dict['loss']['val'] < self.best_error:
                print('Best error.')
                torch.save(ckpt, self.best_error_weights_file)
                self.best_error = iter_dict['loss']['val']
                self.config_dict['best_error'] = self.best_error


        # Saves checkpoint
        torch.save(ckpt, os.path.join(self.save_dir,'iter_{}_weights.pth'.format(ckpt['n_iter'])))
        with open(self.config_file, 'w') as f:
            json.dump(self.config_dict, f)
            
            

            
        
    def load(self):
        """
        Loads an existing checkpoint.
        """
        if os.path.isfile(self.logs_file):
            with open(self.logs_file) as f:
                self.logs_dict = json.load(f)
        if os.path.isfile(self.config_file):
            with open(self.config_file) as f:
                prev_config = json.load(f)
                if 'best_error' in prev_config:
                    self.best_error = prev_config['best_error']
                    self.config_dict['best_error'] = prev_config['best_error']
    
    