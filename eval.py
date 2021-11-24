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
parser.add_argument('--method', required=True, type=str, help='which model to use (MTL, MGDA, PCGrad, gradnorm, MR)')
parser.add_argument('--active_task', default=0, type=int, help='which task to train for STL model (0: classification, 1: OD segmentation, 2: OC segmentation, 3: fovea localization)')

parser.add_argument('--dataroot', default='data/')
parser.add_argument('--split', default='test')
parser.add_argument('--checkpoint_path', default='checkpoints/')
parser.add_argument('--recover', default=True, type=bool, help='recover from a checkpoint')
parser.add_argument('--pretrained', default=True, type=bool)
parser.add_argument('--reco_name', default=None, type=str, help='model to recover from')
parser.add_argument('--reco_type', default='last_checkpoint', type=str, help='which type of recovery (best_error or iter_XXX)')

parser.add_argument('--workers', default=6, type=int)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--optimizer', default='adam', type=str)
parser.add_argument('--per_batch_step', default=False, type=bool, help='optimize tasks altogether or one by one')
parser.add_argument('--one_optim_per_task', default=False, type=bool, help='use one optimizer for all tasks or one per task')
parser.add_argument('--learning_rate', default=1e-4, type=float)
parser.add_argument('--eval_interval', default=400, type=int, help='interval between two validations')
parser.add_argument('--total_epoch', default=500, type=int, help='number of training epochs to proceed')
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
test_metrics = RefugeMetrics(task_groups, opt)
logger = utils.logger(test_metrics.metrics_names)
saver.add_metrics(test_metrics.metrics_names)



# Recover weights, if required
model.initialize(opt, device, model_dir, saver)

# Create datasets and loaders
dataset_path = opt.dataroot
test_data = RefugeDataset2(dataset_path, 
                         split=opt.split)

test_loader = torch.utils.data.DataLoader(
    dataset=test_data,
    batch_size=opt.batch_size,
    num_workers=opt.workers,
    shuffle=False)


# Few parameters
total_epoch = opt.total_epoch
nb_test_batches = len(test_loader)



################
##### Test #####
################
test_avg_losses = np.zeros(n_tasks, dtype=np.float32)
model.eval()
test_dataset = iter(test_loader)
with torch.no_grad():
    for test_k in range(nb_test_batches):
        # Data loading
        test_data, test_gts = test_dataset.next()
        test_data = test_data.to(device)
        test_gts = [test_gts[0].to(device), test_gts[1].to(device), test_gts[2].to(device), [test_gts[3][0].to(device), test_gts[3][1].to(device)]]

        # Logging
        print('Test {}/{} epoch {} iter {}'.format(test_k, nb_test_batches, model.n_epoch, model.n_iter), end=' '*50+'\r')

        # Test step
        task_losses, test_preds = model.test_step(test_data, test_gts, loss_func)
        test_pred_vCDR, test_gt_vCDR = utils.get_vCDRs(test_preds, test_gts)
        clf_preds = torch.from_numpy(model.clf.predict_proba(np.array(test_pred_vCDR).reshape(-1,1))[:,1]).type(torch.float32).to(device)
        test_preds.append(clf_preds)

        # Scoring
        test_avg_losses += task_losses.cpu().numpy() / nb_test_batches
        test_metrics.incr(test_preds, test_gts, test_pred_vCDR, test_gt_vCDR)

# Logging
test_results = test_metrics.result()
print('auc_unet', test_results[0])
print('auc_vcdr', test_results[1])
print('auc', test_results[2])
print('dsc_od', test_results[3])
print('dsc_oc', test_results[4])
print('vCDR_error', test_results[5])
print('fov_error', test_results[6])
