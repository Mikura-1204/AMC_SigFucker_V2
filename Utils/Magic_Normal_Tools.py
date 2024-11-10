import numpy as np
import torch
import os
import json
import sys

# ciallo: 根据设定的条件，动态调整优化器的学习率（learning rate），常用于在训练过程中降低学习率以达到更好的训练效果
class LearningRateAdjuster():
    def __init__(self, initial_lr, patience, lr_decay_rate=0.5, type="type1"):
        self.lr = initial_lr
        self.patience = patience
        self.lr_decay_rate = lr_decay_rate
        assert 0 < self.lr_decay_rate <= 1, "Learning rate decay rate should be between 0 and 1."
        self.type = type

    def _update_lr(self, optimizer):
        self.lr *= self.lr_decay_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.lr
        print(f'Updating learning rate to {self.lr}')

    def rate_decay_with_patience(self, optimizer, patience_count):
        if self.type == "type1":
            if patience_count / self.patience > 0.5:
                self._update_lr(optimizer)
        elif self.type == "type2":
            if patience_count // int(self.patience / 3) == 1:
                self._update_lr(optimizer)

# ciallo: 实现早停机制，用于监控验证集上的损失，当验证损失在多个 epoch 上不再下降时，停止训练，以防止模型过拟合
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, data_parallel=False, times_now=None):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_acc = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.vali_acc_max = np.Inf
        self.delta = delta
        self.save_checkpoint_path = None
        self.data_parallel = data_parallel
        self.times_now = times_now

    def __call__(self, val_loss, vali_acc, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_acc = vali_acc
            self.save_checkpoint(val_loss, vali_acc, model, path)
            save_flag = True
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
            save_flag = False
        else:
            self.best_score = score
            self.best_acc = vali_acc
            self.save_checkpoint(val_loss, vali_acc, model, path)
            self.counter = 0
            save_flag = True
        return self.save_checkpoint_path, save_flag

    def save_checkpoint(self, val_loss, vali_acc, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f})')
            # print(f', and validation acc increased ({self.vali_acc_max:.6f} --> {vali_acc:.6f}).')
            print("saving model...")
        if self.save_checkpoint_path != None:
            os.remove(self.save_checkpoint_path)
        if self.data_parallel:
            self.save_checkpoint_path = path+'/'+'{}_{}_A[{:.4f}]_L[{:.4f}].pth'.format(self.times_now if self.times_now != None else 0,
                                                                                    type(model.module).__name__, vali_acc, val_loss)
        else:
            self.save_checkpoint_path = path+'/'+'{}_{}_A[{:.4f}]_L[{:.4f}].pth'.format(self.times_now if self.times_now != None else 0, 
                                                                                    type(model).__name__, vali_acc, val_loss)
        torch.save(model.state_dict(), self.save_checkpoint_path)
        self.val_loss_min = val_loss
        self.vali_acc_max = vali_acc

# ciallo: 用于对数据进行标准化，即将数据转换为零均值和单位方差，常用于加快模型训练速度并提高模型性能
class StandardScaler():
    def __init__(self, mean=0., std=1.):
        self.mean = mean
        self.std = std
    
    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std

    def inverse_transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data * std) + mean

def load_args(filename):
    with open(filename, 'r') as f:
        args = json.load(f)
    return args

def save_model_structure_in_txt(path, model):
    with open(os.path.join(path, 'model_structure.txt'), 'w') as f:
        sys.stdout = f
        print(model)
        sys.stdout = sys.__stdout__

def string_split(str_for_split, flag):
    str_no_space = str_for_split.replace(' ', '')
    str_split = str_no_space.split(',')
    if flag == "float":
        value_list = [float(x) for x in str_split]
    elif flag == "int":
        value_list = [int(x) for x in str_split]
    else:
        raise ValueError

    return value_list