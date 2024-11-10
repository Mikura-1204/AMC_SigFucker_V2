import argparse
import os
import sys
import torch
import random
import time
from Experiment.Magic_Core_Exp import Exp
from Utils.Magic_Normal_Tools import string_split

parser = argparse.ArgumentParser(description='Automatic Modulation Classification')

parser.add_argument('--dataset', type=str, default='TypeClassify', help='choose dataset')
parser.add_argument('--model', type=str, default='CNN_SigFucker_layer3', help='choose model')
parser.add_argument('--lossFunction', type=str, default='CE', help='choose loss function')

# ciallo: change fucking dataset path (npy file)
parser.add_argument('--dataset_root_path', type=str, default='/media/ubuntu/Elements/Fuck_Data_ShitType5/Fucking_CutNpy2', help='root path of the data file')
parser.add_argument('--data_split', type=str, default='0.8,0.1,0.1', help='train/val/test split, must be ratio, test can be zero')
parser.add_argument('--result_root_path', type=str, default='/media/ubuntu/Elements/Ciallo_SigFucker_ShitType5/Results', help='location to store train results')

parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')

# ciallo: change the training epoch
parser.add_argument('--train_epochs', type=int, default=200, help='train epochs')
parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='optimizer initial learning rate')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--times', type=int, default=1, help='times of experiment')
parser.add_argument('--seed', type=int, default=random.randint(0, 2**32 - 1), help='experiment seed')

parser.add_argument('--n_class', type=int, default=11, help='number of class')

parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', default=False, action='store_true', help='use multiple gpus')
parser.add_argument('--devices', type=str, default='0,1', help='device ids of multile gpus')

args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.device_ids = string_split(args.devices, "int")
    args.gpu = args.device_ids[0]

args.data_split = string_split(args.data_split, "float")
assert sum(args.data_split) == 1 and args.data_split[0] > 0 and args.data_split[1] > 0 and len(args.data_split) <= 3
if len(args.data_split) == 2:
    args.data_split.append(0.0)

# ciallo: 设置类别数
data_parser = {
    "TypeClassify":{"n_class":5},
}

data_info = data_parser[args.dataset]
args.n_class = data_info["n_class"]

print('Args in experiment:')
print(args)

st = time.strftime('%Y%m%d%H%M%S', time.localtime())

exp = Exp(args)
# exp.setup_seed(args.seed)

for i_experiment_time in range(args.times):
    setting = 'experiment_{}_date_{}'.format(args.dataset, st)
    
    print('>>>>>>>start training : {}_times_{}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting, i_experiment_time))
    exp.train(setting, i_experiment_time)
