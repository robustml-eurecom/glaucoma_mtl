import os
import torch
import numpy as np
import argparse
import utils
from losses import RefugeLoss
from dataset import RefugeDataset2
from metrics import *
from saver import Saver

parser = argparse.ArgumentParser()
parser.add_argument('--name', default='test', type=str, help='model name')
parser.add_argument('--method', required=True, type=str, help='which model to use (MTL, STL)')
parser.add_argument('--active_task', default=0, type=int, help='which task to train for STL model (0: classification, 1: OD segmentation, 2: OC segmentation, 3: fovea localization)')

parser.add_argument('--dataroot', default='data/')
parser.add_argument('--checkpoint_path', default='checkpoints/')
parser.add_argument('--recover', default=False, type=bool, help='recover from a checkpoint')
parser.add_argument('--pretrained', default=False, type=bool)
parser.add_argument('--reco_name', default=None, type=str, help='model to recover from')
parser.add_argument('--reco_type', default='last_checkpoint', type=str, help='which type of recovery (best_error or iter_XXX)')

parser.add_argument('--workers', default=6, type=int)
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--optimizer', default='adam', type=str)
parser.add_argument('--one_optim_per_task', default=True, type=bool, help='use one optimizer for all tasks or one per task')
parser.add_argument('--per_batch_step', default=False, type=bool, help='optimize tasks altogether or one by one')
parser.add_argument('--learning_rate', default=1e-4, type=float)
parser.add_argument('--eval_interval', default=5, type=int, help='interval between two validations')
parser.add_argument('--total_epoch', default=300, type=int, help='number of training epochs to proceed')
parser.add_argument('--seed', default=None, type=int)

opt = parser.parse_args()


# Seed
if opt.seed:
    torch.manual_seed(opt.seed)

    
# Saving settings
model_dir = os.path.join(opt.checkpoint_path, opt.name)
os.mkdir(model_dir) if not os.path.isdir(model_dir) else None
saver = Saver(model_dir, args=opt)

# Define model and optimiser
gpu = utils.check_gpu()
device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() else "cpu")
task_groups = [{'ids':0,
                'type':'classif',
                'size': 1},
               {'ids':1,
                'type':'od',
                'size': 1},
               {'ids':2,
                'type':'oc',
                'size': 1},
               {'ids':3,
                'type':'fov',
                'size': 1}]

n_tasks = len(task_groups)
model = utils.select_model(opt, task_groups).to(device)

# Loss and metrics
loss_func = RefugeLoss(task_groups, opt)
train_metrics = RefugeMetrics(task_groups, opt)
val_metrics = RefugeMetrics(task_groups, opt)
logger = utils.logger(train_metrics.metrics_names)
saver.add_metrics(train_metrics.metrics_names)



# Recover weights, if required
model.initialize(opt, device, model_dir, saver)

# Create datasets and loaders
dataset_path = opt.dataroot
train_data = RefugeDataset2(dataset_path, 
                          split='train')
val_data = RefugeDataset2(dataset_path, 
                        split='val')

train_loader = torch.utils.data.DataLoader(
    dataset=train_data,
    batch_size=opt.batch_size,
    num_workers=opt.workers,
    shuffle=True,
    drop_last=True)

val_loader = torch.utils.data.DataLoader(
    dataset=val_data,
    batch_size=1,
    num_workers=opt.workers,
    shuffle=False)


# Few parameters
total_epoch = opt.total_epoch
nb_train_batches = len(train_loader)
nb_val_batches = len(val_loader)
train_avg_losses = np.zeros(n_tasks, dtype=np.float32)

# Iterations
while model.n_epoch < total_epoch:
    ####################
    ##### Training #####
    ####################
    model.train()
    train_dataset = iter(train_loader)
    train_vCDRs = []
    train_classif_labs = []
    for train_k in range(nb_train_batches):
        # Data loading
        train_data, train_gts = train_dataset.next()
        train_data = train_data.to(device)
        train_gts = [train_gts[0].to(device), train_gts[1].to(device), train_gts[2].to(device), [train_gts[3][0].to(device), train_gts[3][1].to(device)]]
        train_classif_labs += train_gts[0].cpu().numpy().tolist()
        
        # Train step
        task_losses, train_preds = model.train_step(train_data, train_gts, loss_func)
        train_pred_vCDR, train_gt_vCDR = utils.get_vCDRs(train_preds, train_gts)
        train_preds.append(train_preds[0])
        
        # Scoring
        train_avg_losses += task_losses.cpu().numpy()
        train_metrics.incr(train_preds, train_gts, train_pred_vCDR, train_gt_vCDR)
        train_vCDRs += train_gt_vCDR.tolist()
        
        # Logging
        print('Epoch {}, iter {}/{}, losses ({:.2f}, {:.2f}, {:.2f}, {:.2f})'.format(model.n_epoch, train_k, nb_train_batches, task_losses[0], task_losses[1], task_losses[2], task_losses[3]), end=' '*50+'\r')
        
        

    ######################
    ##### Validation #####
    ######################
    if (model.n_epoch+1)%opt.eval_interval == 0:
        train_vCDRs = np.array(train_vCDRs).reshape(-1,1)
        model.train_clf(train_vCDRs, np.array(train_classif_labs))
        train_vCDRs = []
        train_classif_labs = []
        val_avg_losses = np.zeros(n_tasks, dtype=np.float32)
        model.eval()
        val_dataset = iter(val_loader)
        with torch.no_grad():
            for val_k in range(nb_val_batches):
                # Data loading
                val_data, val_gts = val_dataset.next()
                val_data = val_data.to(device)
                val_gts = [val_gts[0].to(device), val_gts[1].to(device), val_gts[2].to(device), [val_gts[3][0].to(device), val_gts[3][1].to(device)]]

                # Logging
                print('Eval {}/{} epoch {} iter {}'.format(val_k, nb_val_batches, model.n_epoch, model.n_iter), end=' '*50+'\r')

                # Test step
                task_losses, val_preds = model.test_step(val_data, val_gts, loss_func)
                val_pred_vCDR, val_gt_vCDR = utils.get_vCDRs(val_preds, val_gts)
                clf_preds = torch.from_numpy(model.clf.predict_proba(np.array(val_pred_vCDR).reshape(-1,1))[:,1]).type(torch.float32).to(device)
                val_preds.append(clf_preds)

                # Scoring
                val_avg_losses += task_losses.cpu().numpy() / nb_val_batches
                val_metrics.incr(val_preds, val_gts, val_pred_vCDR, val_gt_vCDR)


        # Logging
        train_avg_losses /= (nb_train_batches*opt.eval_interval)
        train_results = train_metrics.result()
        val_results = val_metrics.result()
        logger.log(model.n_epoch, 
                   model.n_iter, 
                   train_results, 
                   val_results, 
                   train_avg_losses.sum(), 
                   val_avg_losses.sum())
        saver.log(model, 
                  task_groups, 
                  model.n_epoch, 
                  model.n_iter, 
                  train_results, 
                  val_results, 
                  train_avg_losses, 
                  val_avg_losses,
                  model.optimizer)
        train_metrics.reset()
        val_metrics.reset()
        model.train()

    
    # Update epoch and LR
    model.n_epoch += 1
    

